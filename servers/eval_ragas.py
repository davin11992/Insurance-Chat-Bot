import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# gpt-5-mini의 temperature 비지원 문제 우회를 위한 openai 패치
import openai
orig_create = openai.resources.chat.completions.Completions.create
def safe_create(self, *args, **kwargs):
    if "temperature" in kwargs:
        del kwargs["temperature"]
    return orig_create(self, *args, **kwargs)
openai.resources.chat.completions.Completions.create = safe_create

orig_acreate = openai.resources.chat.completions.AsyncCompletions.create
async def safe_acreate(self, *args, **kwargs):
    if "temperature" in kwargs:
        del kwargs["temperature"]
    return await orig_acreate(self, *args, **kwargs)
openai.resources.chat.completions.AsyncCompletions.create = safe_acreate


# 1. 기존 main.py 모듈 및 LangChain 구성요소 로드
import main
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# .env 파일 로드
load_dotenv()

print("1. RAG 에이전트 인프라 초기화 중...")
# main의 전역 설정값을 기반으로 LLM 및 임베딩 모델 수동 초기화
main.llm = ChatOpenAI(
    model=main.GPT_MODEL,
    api_key=main.OPENAI_API_KEY,
    base_url=main.BASE_URL
)
main.embeddings = OpenAIEmbeddings(
    model=main.EMBEDDING_MODEL_NAME,
    api_key=main.OPENAI_API_KEY,
    base_url=main.BASE_URL
)

# ChromaDB 및 템플릿 초기화 (Graph 컴파일은 몽키 패치 이후에 진행)
main.init_vectorstore()
main.prompt_template, main.prompt_template_general = main.init_prompt_templates()

# 평가 대상 약관 문서 경로: 환경변수 EVAL_DOC 우선, 없으면 ./data 안의 첫 문서 사용
def _find_eval_doc() -> Path | None:
    env_doc = os.getenv("EVAL_DOC")
    if env_doc and Path(env_doc).exists():
        return Path(env_doc)
    data_dir = Path("./data")
    if data_dir.exists():
        candidates = sorted(
            p for p in data_dir.iterdir()
            if p.suffix.lower() in {".pdf", ".md", ".txt"}
        )
        if candidates:
            return candidates[0]
    return None

EVAL_DOC_PATH = _find_eval_doc()

# 평가 결과 산출물은 저장소에 포함되도록 docs/ 에 저장
RESULT_DIR = Path(__file__).resolve().parent.parent / "docs"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 평가 시작 전 약관 문서가 DB에 없다면 자동으로 적재
print("1-2. 평가를 위한 약관 문서 적재 검사...")
try:
    # 컬렉션 내 문서가 비어있는지 유사도 임의 검색으로 확인
    sample_search = main.vectorstore.similarity_search("보험", k=1)
    if not sample_search:
        if EVAL_DOC_PATH is not None:
            print(f"-> DB가 비어있음을 감지했습니다. '{EVAL_DOC_PATH.name}' 문서를 분할하여 적재합니다...")
            docs = main.extract_documents(EVAL_DOC_PATH)
            chunk_docs = main.chunk_documents(docs)
            main.add_to_db(chunk_docs)
            print(f"-> {len(chunk_docs)}개 청크 적재 완료.")
        else:
            print("🚨 경고: ./data 안에 약관 문서(.pdf/.md/.txt)가 없습니다. 먼저 약관을 업로드하거나 EVAL_DOC 환경변수를 지정하세요.")
    else:
        print("-> DB에 기존 문서 데이터가 존재합니다. 기존 데이터를 사용하여 평가합니다.")
except Exception as e:
    print(f"문서 사전 적재 확인 중 예외 발생 (무시하고 계속 진행): {e}")

# 최종적으로 Graph를 빌드하여 컴파일
main.rag_workflow = main.build_rag_graph()
print("-> RAG 워크플로우 컴파일 완료.")



import json

# 2. Ragas 평가용 테스트 케이스 정의 (질문 및 모범 답안)
test_cases_file = Path("./test_cases.json")
test_cases = []

if test_cases_file.exists():
    print(f"-> 기존 생성된 테스트 케이스 파일({test_cases_file})을 로드합니다.")
    try:
        with open(test_cases_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    except Exception as le:
        print(f"-> 테스트 케이스 파일 로드 중 오류 발생: {le}")

if not test_cases:
    print("-> 테스트 케이스 캐시가 없거나 로드에 실패했습니다. Ragas Testset Generator로 5개의 테스트 케이스를 자동 생성합니다...")
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.testset.synthesizers.generate import TestsetGenerator

        guide_path = EVAL_DOC_PATH
        if guide_path is not None and guide_path.exists():
            docs = main.extract_documents(guide_path)
            chunk_docs = main.chunk_documents(docs)

            # Ragas 래퍼 클래스로 LLM 및 Embeddings 모델 포장
            generator_llm = LangchainLLMWrapper(main.llm)
            generator_embeddings = LangchainEmbeddingsWrapper(main.embeddings)

            generator = TestsetGenerator(
                llm=generator_llm,
                embedding_model=generator_embeddings
            )
            # Ragas가 한국어로 질문과 정답을 생성하도록 프롬프트 번역 및 언어 로컬라이징 적용
            generator.adapt_to_language("ko")


            from ragas.run_config import RunConfig
            run_config = RunConfig(
                max_workers=2,
                max_retries=10,
                timeout=180
            )

            print("-> LLM 기반 테스트셋 생성 중 (1~2분 정도 소요될 수 있습니다)...")
            testset = generator.generate_with_langchain_docs(
                documents=chunk_docs,
                testset_size=5,
                run_config=run_config,
                with_debugging_logs=False
            )

            df_testset = testset.to_pandas()
            for _, row in df_testset.iterrows():
                # Ragas 버전별 컬럼명 차이 대응 (question/ground_truth vs user_input/reference)
                q = row.get("user_input") or row.get("question")
                gt = row.get("reference") or row.get("ground_truth")
                if q and gt:
                    test_cases.append({
                        "question": q,
                        "ground_truth": gt
                    })

            # 생성된 테스트 케이스를 파일에 저장
            if test_cases:
                with open(test_cases_file, "w", encoding="utf-8") as f:
                    json.dump(test_cases, f, ensure_ascii=False, indent=4)
                print(f"-> {len(test_cases)}개의 테스트 케이스를 생성하여 '{test_cases_file}'에 캐싱 완료.")
        else:
            print("🚨 경고: 평가 대상 약관 문서가 없어 테스트셋 생성을 생략합니다.")
    except Exception as ge:
        print(f"🚨 Ragas Testset 생성 실패 (기본 Fallback 질문으로 대체): {ge}")

# 생성 및 캐싱 실패 시 최종 Fallback 질문 세트 지정
if not test_cases:
    print("-> 기본 정의된 3개의 테스트 케이스를 사용합니다.")
    test_cases = [
        {
            "question": "보험금을 청구할 때 어떤 서류를 제출해야 하나요?",
            "ground_truth": "회사양식 청구서, 신분증(사진이 붙은 정부기관 발행 신분증), 기타 보험금 수령에 필요하여 제출하는 서류를 제출하고 청구해야 합니다."
        },
        {
            "question": "보험금은 청구 후 며칠 안에 지급되나요?",
            "ground_truth": "청구 서류를 접수한 날부터 3영업일 이내에 지급하며, 지급사유의 조사·확인이 필요한 경우에는 접수 후 10영업일 이내에 지급합니다."
        },
        {
            "question": "공시이율의 최저보증이율은 얼마인가요?",
            "ground_truth": "전환 후 10년 이내에는 연복리 1.0%, 10년을 초과한 경우에는 연복리 0.5%를 최저보증합니다."
        }
    ]


# 3. RAG 시스템을 호출하여 평가에 필요한 데이터(답변, 검색된 컨텍스트) 수집
print("\n2. 테스트 케이스에 대한 RAG 응답 수집 중...")
eval_data = []

for case in test_cases:
    question = case["question"]
    ground_truth = case["ground_truth"]

    print(f"질문 수행 중: {question}")
    # RAG 워크플로우 직접 실행
    initial_state = {
        "question": question,
        "use_rag": True,
        "context": "",
        "source": "",
        "grounded": False
    }
    final_state = main.rag_workflow.invoke(initial_state)

    # RAG가 찾아온 원본 문장 리스트 확보
    retrieved_contexts = [final_state["context"]] if final_state.get("context") else []

    eval_data.append({
        "user_input": question,
        "response": final_state["answer"],
        "retrieved_contexts": retrieved_contexts,
        "reference": ground_truth
    })

# 4. Ragas EvaluationDataset 구축 및 평가 실행
print("\n3. Ragas 평가 시작 (LLM 심사 구동)...")
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextRecall, ContextPrecision

# Ragas 0.4 규격에 맞게 데이터셋 로드
dataset = EvaluationDataset.from_list(eval_data)

# 평가에 사용할 지표 구성 (4대 지표)
metrics = [
    Faithfulness(llm=main.llm),
    AnswerRelevancy(llm=main.llm),
    LLMContextRecall(llm=main.llm),
    ContextPrecision(llm=main.llm)
]

# Ragas 평가 실행
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=main.llm,
    embeddings=main.embeddings
)

# 4-2. 거절(hallucination 방지) 케이스 별도 점검 — 약관에 없는 질문에 지어내지 않는지 확인 (F206)
print("\n3-2. 거절 케이스 점검 (약관에 없는 질문)...")
refusal_file = Path("./test_cases_refusal.json")
refusal_pass = refusal_total = 0
refusal_rows = []
if refusal_file.exists():
    refusal_cases = json.loads(refusal_file.read_text(encoding="utf-8"))
    REFUSAL_KEYWORDS = ("약관에 명시되어 있지 않", "약관에서 찾지 못", "찾을 수 없", "명시되어 있지 않")
    for rc in refusal_cases:
        q = rc["question"]
        st = main.rag_workflow.invoke(
            {"question": q, "use_rag": True, "context": "", "source": "", "grounded": False}
        )
        ans = st["answer"]
        ok = any(k in ans for k in REFUSAL_KEYWORDS)
        refusal_total += 1
        refusal_pass += int(ok)
        refusal_rows.append((q, "PASS" if ok else "FAIL", ans.replace("\n", " ")[:80]))
        print(f"  [{'PASS' if ok else 'FAIL'}] {q}")
    print(f"  -> 거절 정확도: {refusal_pass}/{refusal_total}")
else:
    print("  (test_cases_refusal.json 없음 — 건너뜀)")

# 5. 결과 시각화 및 저장
print("\n================ Ragas 평가 결과 ================")
print(results)
print(f"거절 케이스 정확도: {refusal_pass}/{refusal_total}" if refusal_total else "거절 케이스: 없음")
print("=================================================")

# 판다스 데이터프레임으로 변환하여 상세 결과 출력 및 CSV/MD 저장
df_results = results.to_pandas()
print("\n[상세 리포트]")
print(df_results[["user_input", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]])

# CSV 저장 (docs/)
output_csv = str(RESULT_DIR / "ragas_eval_results.csv")
df_results.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n평가 상세 결과가 '{output_csv}' 파일로 저장되었습니다.")

# MD 저장 (누적 추가)
from datetime import datetime
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
output_md = str(RESULT_DIR / "ragas_eval_history.md")

# 평균값 계산
avg_faithfulness = df_results["faithfulness"].mean()
avg_relevancy = df_results["answer_relevancy"].mean()
avg_recall = df_results["context_recall"].mean()
avg_precision = df_results["context_precision"].mean()

# 마크다운 내용 구성
md_content = f"""
## RAG 성능 평가 결과 (실행 시각: {now_str})

### 1. 평균 점수 요약
| Faithfulness (충실도) | Answer Relevancy (관련성) | Context Recall (재현율) | Context Precision (정밀도) | 거절 정확도 |
| :---: | :---: | :---: | :---: | :---: |
| {avg_faithfulness:.4f} | {avg_relevancy:.4f} | {avg_recall:.4f} | {avg_precision:.4f} | {refusal_pass}/{refusal_total} |

### 2. 상세 질문별 리포트
| 질문 (user_input) | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
| :--- | :---: | :---: | :---: | :---: |
"""

for _, row in df_results.iterrows():
    # Ragas 평가지표 중 None이나 NaN 값이 있을 수 있으므로 안전하게 처리
    f_val = row.get("faithfulness", 0.0)
    r_val = row.get("answer_relevancy", 0.0)
    rc_val = row.get("context_recall", 0.0)
    p_val = row.get("context_precision", 0.0)

    f_str = f"{f_val:.4f}" if pd.notna(f_val) else "N/A"
    r_str = f"{r_val:.4f}" if pd.notna(r_val) else "N/A"
    rc_str = f"{rc_val:.4f}" if pd.notna(rc_val) else "N/A"
    p_str = f"{p_val:.4f}" if pd.notna(p_val) else "N/A"

    md_content += f"| {row['user_input']} | {f_str} | {r_str} | {rc_str} | {p_str} |\n"

md_content += "\n---\n"

# 파일에 누적(Append)하여 쓰기
file_exists = os.path.exists(output_md)
with open(output_md, "a", encoding="utf-8") as f:
    if not file_exists:
        f.write("# Ragas RAG 성능 평가 누적 기록\n\n")
    f.write(md_content)

print(f"평가 결과가 누적 기록 파일 '{output_md}'에 추가되었습니다.")

