# 보험 약관 Q&A 봇

보험 약관 PDF를 업로드하면, 그 내용을 근거로 질문에 답하는 RAG 챗봇입니다.
데모·평가 문서로는 **삼성생명 플러스연금전환특약(무배당) 약관**을 사용합니다.
(SSAFY JAVA 대전 4반 — 이다빈 / 정소현)

> 제출 파일명: `ChatBot_보험약관_대전_4반_이다빈_정소현.zip`

## 개요

- 사용자가 보험 약관(PDF/MD/TXT)을 업로드 → 문서를 청킹하여 벡터 DB(Chroma)에 적재
- 질문이 들어오면 유사한 약관 조항을 검색(RAG)하여 LLM이 근거 기반으로 답변 (가능하면 조항 번호 인용)
- 약관에 없는 내용은 "해당 내용은 약관에 명시되어 있지 않습니다"로 답하고, 항상 약관 원문·보험사 상담 확인 문구를 덧붙임
- Ragas로 RAG 답변 품질(충실도·관련성·재현율·정밀도)을 평가 ([servers/test_cases.json](servers/test_cases.json) 12문항) + 약관에 없는 질문을 지어내지 않고 거절하는지 별도 점검 ([servers/test_cases_refusal.json](servers/test_cases_refusal.json) 3문항)

## 구조

```
clients/            프론트엔드 (순수 HTML/CSS/JS 채팅 UI)
  index.html
  script.js
  style.css
  sample_data/      학습시킬 약관 문서 위치 (커밋 제외)
servers/            백엔드 (FastAPI + LangGraph + Chroma)
  main.py           API 서버 / RAG 워크플로우
  eval_ragas.py     Ragas 기반 성능 평가 스크립트
  test_cases.json          평가용 질문/모범답안 12문항
  test_cases_refusal.json  약관 밖 질문 3문항 (거절 점검용)
  .env.example      환경변수 템플릿 (실제 .env 는 커밋 제외)
  requirements.txt
docs/
  고찰_보고서.md            제출용 고찰 보고서
  img1~4.png               실행 화면 / 평가결과 캡처
  ragas_eval_history.md    Ragas 평가 누적 기록 (eval_ragas.py 가 생성)
  ragas_eval_results.csv   Ragas 평가 상세 결과
```

### 백엔드 아키텍처

- **FastAPI**: `/chat`, `/upload`, `/reset-db` 엔드포인트
- **LangGraph**: `retrieve → generate` 2노드 워크플로우, `use_rag` 플래그로 검색 우회 분기
- **Chroma**: 로컬 벡터 스토어 (`collection_name="insurance_docs"`, 서버 종료 시 자동 정리)
- **OpenAI**: ChatOpenAI + OpenAIEmbeddings (`base_url` 커스텀 지원)

## 실행 방법

### 1. 백엔드

```bash
cd servers
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # 후 API_KEY 등 값 채우기
```

```bash
python main.py          # http://localhost:8000
```

### 2. 프론트엔드

`clients/index.html` 를 Live Server 등으로 실행 (127.0.0.1 / localhost 모든 포트 CORS 허용).

### 3. 사용

1. 우측 상단 **문서 추가** → 보험 약관 PDF 업로드 (데모: 플러스연금전환특약 약관)
2. 입력창에 질문 입력, **RAG** 체크박스 on → 약관 근거 답변
3. **DB 초기화** 로 적재된 문서 전체 삭제

## 성능 평가 (Ragas)

```bash
cd servers
# 평가 대상 약관을 servers/data/ 에 두거나 EVAL_DOC 환경변수로 지정
python eval_ragas.py
```

- `test_cases.json` 이 있으면 그대로 사용, 없으면 약관에서 자동 생성 시도 후 기본 fallback 질문 사용
- 결과: [docs/ragas_eval_results.csv](docs/ragas_eval_results.csv), 누적 기록 [docs/ragas_eval_history.md](docs/ragas_eval_history.md)
- 지표: Faithfulness / Answer Relevancy / Context Recall / Context Precision + 거절 정확도(N/3)

### 최종 결과 (3회 튜닝)

| Faithfulness | Answer Relevancy | Context Recall | Context Precision | 거절 정확도 |
| :---: | :---: | :---: | :---: | :---: |
| 0.98 | 0.71 | 1.00 | 1.00 | 3/3 |

개선 과정과 지표 해석은 [docs/고찰_보고서.md](docs/고찰_보고서.md) 3절 참조.

## 실행 화면

| RAG ON (약관 근거 답변) | RAG OFF (같은 질문, 일반 GPT) | 약관 밖 질문 → 거절 |
|---|---|---|
| ![RAG ON](docs/img1.png) | ![RAG OFF](docs/img2.png) | ![거절](docs/img3.png) |

- **RAG ON**: 약관 제4조·제2조를 근거로 답하고 `📄 약관 근거` 배지 표시
- **RAG OFF**: 일반 상식으로 답하나 이 약관의 실제 내용과 다름, `⚠️ 일반 AI 응답` 배지
- **약관 밖 질문**: "해당 내용은 약관에 명시되어 있지 않습니다" 로 거절 (환각 방지)
