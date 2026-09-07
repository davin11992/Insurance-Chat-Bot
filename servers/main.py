import os
import warnings
warnings.filterwarnings("ignore")


# transformers 및 huggingface 캐시 관련 경고 방지
if "HF_HOME" not in os.environ and "TRANSFORMERS_CACHE" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import TypedDict, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# LangGraph Core 임포트
from langgraph.graph import StateGraph, START, END

# .env 로드
load_dotenv()

# ==========================================
# [설정 영역]
# ==========================================
# ==========================================
# TODO: 01 .env에서 사용자 환경에 맞게 수정하세요. 
# ==========================================
OPENAI_API_KEY = os.getenv("API_KEY", "키를 입력하세요.")   # .env의 API_KEY 로드
BASE_URL = os.getenv("base_url")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 800               # 텍스트를 분할할 글자 수 (약관 조항이 길어 넉넉하게)
OVERLAP = 150                  # 청크 간 겹치게 할 글자 수 (문맥 보존용)
RELEVANCE_THRESHOLD = 0.2      # 유사도 점수가 이 값 미만인 청크는 근거로 사용하지 않음 (F205/F206)
RETRIEVE_K = 5                 # 유사도 검색으로 가져올 후보 청크 수
CHROMA_DB_PATH = "./chroma_data" # ChromaDB가 파일로 저장될 로컬 디렉토리 경로

# ==========================================
# [전역 변수] 서버 실행 중 메모리에 유지될 객체들
# ==========================================
llm = None                  # OpenAI 연결 클라이언트
embeddings = None           # 임베딩 모델 (텍스트 -> 벡터 변환)
vectorstore = None          # Chroma VectorStore 객체
prompt_template = None      # ChatPromptTemplate 객체 (RAG 전용)
prompt_template_general = None  # ChatPromptTemplate 객체 (일반 대화용)
rag_workflow = None         # 컴파일된 LangGraph RAG 에이전트 애플리케이션


# ==========================================
# [LangGraph State 정의]
# ==========================================
class RAGState(TypedDict):
    question: str         # 사용자 질문
    context: str          # 유사도 검색을 통해 조립된 본문 컨텍스트
    answer: str           # LLM이 생성한 최종 답변
    source: str           # 정밀 답변 출처 태그
    use_rag: bool         # RAG 사용 여부 제어 플래그
    grounded: bool        # 임계값을 넘는 약관 근거로 답변했는지 여부 (F206: 문서 참고 여부)

# ==========================================
# [LangGraph Node 정의]
# ==========================================
def retrieve_node(state: RAGState) -> RAGState:
    """[Retrieve Node]: 질문(question)을 받아 관련 문서를 책장에서 검색한 뒤 컨텍스트(context)를 조립합니다."""
    question = state["question"]

    # 유사도 스코어(0.0~1.0)와 함께 검색 실행
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        question,
        k=RETRIEVE_K,
        score_threshold=0.00
    )

    # 터미널 디버그용 출력 (임계값 채택 여부 포함)
    print("\n--- [약관 검색 결과] ---")
    for i, (doc, score) in enumerate(docs_with_scores):
        adopted = "O" if score >= RELEVANCE_THRESHOLD else "X"
        print(f"[{i+1}위] 채택 {adopted} | 점수: {score:.4f} | 출처: {doc.metadata.get('source', '알 수 없음')}")
        print(f"   내용: {doc.page_content.strip()[:120]}...")
    print("----------------------------\n")

    # F205: 유사도 임계값을 만족하는 청크만 근거로 사용
    relevant_docs = [doc for doc, score in docs_with_scores if score >= RELEVANCE_THRESHOLD]

    # F206: 임계값을 넘는 근거 청크가 없으면 grounded=False로 표시 (답변 분기는 generate_node에서)
    if not relevant_docs:
        return {
            "context": "",
            "source": "일반 지식 (약관에서 근거를 찾지 못함)",
            "grounded": False
        }

    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    source_tag = ", ".join(sorted({doc.metadata.get("source", "알 수 없음") for doc in relevant_docs}))

    return {
        "context": context,
        "source": source_tag,
        "grounded": True
    }

NO_EVIDENCE_ANSWER = (
    "질문하신 내용을 업로드된 약관에서 찾지 못했습니다. "
    "약관에 없는 내용이거나, 다른 표현(조항 용어 등)으로 다시 질문하시면 찾을 수 있습니다."
)

# 프론트엔드에 별도로 전달하는 고정 면책 문구 (답변 본문에는 포함하지 않아 평가 지표를 왜곡하지 않음)
DISCLAIMER = "정확한 내용은 보험증권·약관 원문 및 보험사 상담을 통해 확인하세요."

def generate_node(state: RAGState) -> RAGState:
    """[Generate Node]: 검색된 참고문서(context)를 기반으로 사용자 질문(question)에 답변을 생성합니다."""
    question = state["question"]
    context = state.get("context", "")
    grounded = state.get("grounded", False)
    use_rag = state.get("use_rag", False)

    if grounded and context:
        # 임계값을 넘는 약관 근거가 있음 → RAG 프롬프트
        response = (prompt_template | llm).invoke({"context": context, "question": question})
        return {"answer": response.content}

    if use_rag:
        # 약관 기반 답변을 요청했으나 근거를 찾지 못함 → 일반 지식으로 지어내지 않고 명확히 안내 (F206)
        return {"answer": NO_EVIDENCE_ANSWER}

    # RAG 토글이 꺼져 있음 → 일반 대화 프롬프트
    response = (prompt_template_general | llm).invoke({"question": question})
    return {"answer": response.content}

# ==========================================
# [LangGraph RAG 워크플로우 빌드]
# ==========================================
def route_by_rag_flag(state: RAGState) -> str:
    """use_rag 플래그에 따라 다음 실행할 노드를 결정하는 조건부 라우터 함수"""
    if state.get("use_rag", True):
        return "retrieve"
    return "generate"

def build_rag_graph():
    # TODO: 03. workflow를 구성하고 compile 해서 반환하시오.
    #  workflow 생성
    workflow = StateGraph(RAGState)

    #  노드 추가
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    #  조건부 엣지 정의 (START에서 use_rag 분기에 따라 retrieve 혹은 generate로 바로 이동)
    workflow.add_conditional_edges(
        START,
        route_by_rag_flag,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    #  나머지 엣지 정의
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    #  컴파일 후 반환
    return workflow.compile()

    # END

# ==========================================
# [헬퍼 함수] 문서 추출, 청킹, DB 적재
# ==========================================
def extract_documents(file_path: Path) -> list[Document]:
    """파일에서 Document 객체 리스트를 추출합니다. (PyMuPDFLoader 또는 TextLoader 사용)"""
    try:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            loader = PyMuPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        docs = loader.load()
        # 모든 문서의 metadata에 source 정보로 파일명 저장
        for doc in docs:
            doc.metadata["source"] = file_path.name
        return docs
    except Exception as e:
        print(f"문서 추출 오류: {e}")
    return []

def chunk_documents(documents: list[Document], size=CHUNK_SIZE, overlap=OVERLAP) -> list[Document]:
    """LangChain의 RecursiveCharacterTextSplitter를 사용하여 Document 리스트를 분할합니다."""
    if not documents:
        return []
    # TODO: 04. RecursiveCharacterTextSplitter를 구성해보자.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,             # 청크 하나의 최대 글자 수
        chunk_overlap=overlap,       # 인접 청크 간 겹치는 글자 수 (문맥 보존)
        separators=["\n\n", "\n", ". ", " ", ""],  # 큰 단위부터 재귀적으로 분할
        length_function=len,
    )
    # Document 리스트를 분할하면서 metadata(source 등)는 그대로 승계됨
    return splitter.split_documents(documents)

    # END


def add_to_db(documents: list[Document]) -> int:
    """분할된 Document 객체 리스트를 Chroma DB에 추가합니다."""
    if not documents:
        return 0

    vectorstore.add_documents(documents)
    return len(documents)

def init_vectorstore():
    """Chroma VectorStore 객체를 생성/초기화합니다."""
    global vectorstore
    vectorstore = Chroma(
        collection_name="insurance_docs",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    return vectorstore

def init_prompt_templates():
    """RAG용 및 일반 대화용 프롬프트 템플릿을 생성합니다."""
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 보험 약관을 안내하는 AI 어시스턴트입니다.\n"
            "아래의 [참고 약관] 내용만을 근거로 사용자의 질문에 답변하세요.\n"
            "질문의 핵심에 대해 완결된 한두 문장으로 서술형으로 답하고, 핵심 수치·조건은 문장 안에 포함하세요.\n"
            "군더더기 문장이나 일반적인 면책 안내는 붙이지 마세요. 근거 조항은 문장 맨 끝에 (제0조) 형태로 짧게만 표기하세요.\n"
            "여러 항목을 나열할 때만 목록을 사용하세요.\n"
            "[참고 약관]에서 답을 찾을 수 없으면 '해당 내용은 약관에 명시되어 있지 않습니다.' 라고만 답하세요.\n\n"
            "[참고 약관]\n\n"
            "{context}"
        )),
        ("user", "{question}")
    ])

    general_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 보험에 대한 일반적인 정보를 안내하는 친절한 AI 어시스턴트입니다.\n"
            "일반적인 보험 상식 수준에서 간결하게 답변하되, 구체적인 조건은 상품·약관마다 다를 수 있음을 한 문장으로만 안내하세요."
        )),
        ("user", "{question}")
    ])
    return rag_prompt, general_prompt

# ==========================================
# [FastAPI 수명주기 초기화]
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings, llm, vectorstore, prompt_template, prompt_template_general, rag_workflow
    print("1. AI 모델 및 클라이언트 초기화 중...")

    # LangChain ChatOpenAI 초기화 (base_url 지원)
    llm = ChatOpenAI(
        model=GPT_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL
    )
    # embedding 모델 초기화
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL
    )

    # ChromaDB 저장소 초기화 및 연결
    init_vectorstore()

    # 프롬프트 템플릿 초기화
    prompt_template, prompt_template_general = init_prompt_templates()

    # LangGraph 워크플로우 구성 및 컴파일
    print("2. LangGraph RAG 워크플로우 빌드 중...")
    rag_workflow = build_rag_graph()
    print("준비 완료! 서버가 시작되었습니다.")
    yield
    print("서버가 종료됩니다.")
    # [종료 시 클린업] Chroma SQLite 연결을 닫고 물리 폴더 삭제
    if vectorstore is not None:
        if hasattr(vectorstore, "_client"):
            try:
                vectorstore._client.close()
            except Exception:
                pass
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"'{CHROMA_DB_PATH}' 폴더가 자동 정리 및 삭제되었습니다.")

def setup_cors(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        # TODO: 02. CORS 설정을 적용합니다. (127.0.0.1의 모든 포트 허용)
        # Live Server가 127.0.0.1 또는 localhost 중 어느 쪽으로 열리든 허용
        allow_origin_regex=r"https?://(127\.0\.0\.1|localhost)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # END
    )

app = FastAPI(lifespan=lifespan)
setup_cors(app)

# API 요청용 스키마
class ChatReq(BaseModel):
    message: str
    use_rag: bool = False     # 기본값으로 RAG 사용을 활성화

# ==========================================
# [CORS 테스트 API] Simple vs Preflight 실습
# ==========================================
# 1. Simple Request - form 파라미터 (CORS 프리플라이트 대상 아님)
@app.post("/simpleparam")
def simple_param(message: str = Form(...)):
    print("message: ", message)
    return {
        "type": "simple_request",
        "message": f"받은 메시지: {message}",
        "preflight": "불필요 (application/x-www-form-urlencoded)"
    }

# 2. Preflight 필요 - JSON (CORS 프리플라이트 강제 대상)
@app.post("/simplejson")
def simple_json(req: ChatReq):
    print("message: ", req)
    return {
        "type": "preflight_request",
        "message": f"받은 메시지: {req.message}",
        "preflight": "필요 (application/json)"
    }

# ==========================================
# 통합 채팅 (LangGraph RAG 워크플로우 호출)
# ==========================================
@app.post("/chat")
def chat(req: ChatReq):
    initial_state = {
        "question": req.message,
        "use_rag": req.use_rag,
        "context": "",
        "source": "",
        "grounded": False
    }

    final_state = rag_workflow.invoke(initial_state)
    grounded = final_state.get("grounded", False)

    return {
        "answer": final_state["answer"],
        "grounded": grounded,                                   # F206: 문서 참고 여부
        "source": final_state.get("source") if grounded else None,
        "disclaimer": DISCLAIMER
    }

# ==========================================
# [API 2] 파일 업로드 (동적 문서 추가)
# ==========================================
@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    allowed_extensions = {".txt", ".md", ".pdf"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return {
            "success": False,
            "message": f"지원하지 않는 파일 형식입니다. ({', '.join(allowed_extensions)} 만 허용)"
        }
    try:
        # 파일을 data 폴더에 백업 저장
        save_path = DATA_DIR / file.filename
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 백업된 파일 경로를 전달하여 Document 객체 리스트 추출
        docs = extract_documents(file_path=save_path)
        if not docs:
           return {"success": False, "message": "파일 내용이 없거나 읽을 수 없습니다."}

        # Document 청킹 및 DB 벡터 적재
        chunk_docs = chunk_documents(docs)

        # vdctor store에 데이터 저장
        chunks_added = add_to_db(chunk_docs)
        return {
            "success": True,
            "message": f"'{file.filename}' 업로드 완료! ({chunks_added}개 청크 추가)",
            "chunks_added": chunks_added
        }
    except Exception as e:
        return {"success": False, "message": f"파일 처리 중 오류 발생: {str(e)}"}

# ==========================================
# [API 3] DB 초기화 (Chroma DB 비우기)
# ==========================================
@app.post("/reset-db")
def reset_db():
    try:
        if vectorstore is not None:
            # 컬렉션을 지우고 새로 만드는 대신, 기존 컬렉션 안의 모든 문서 ID를 삭제
            all_ids = vectorstore.get()["ids"]
            if all_ids:
                vectorstore.delete(ids=all_ids)

        return {"success": True, "message": "Chroma DB 데이터가 완전히 초기화되었습니다."}
    except Exception as e:
        return {"success": False, "message": f"DB 초기화 중 오류 발생: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
