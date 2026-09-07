// script.js - 보험 약관 Q&A 챗봇
// RAG 토글이 켜져 있으면 업로드된 보험 약관을 근거로 답변합니다.

function sendMessage() {
    const inputElement = document.getElementById("user-input");
    const ragToggle = document.getElementById("rag-toggle");
    const userMessage = inputElement.value;

    if (userMessage === "") return;

    addMessage("user", userMessage);
    inputElement.value = "";

    // 체크박스가 체크되어 있으면 use_rag를 true로 설정하여 단일 /chat API 호출
    const useRag = ragToggle ? ragToggle.checked : false;
    const endpoint = "http://localhost:8000/chat";

    fetch(endpoint, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            message: userMessage,
            use_rag: useRag
        })
    })
    .then(function(response) {
        return response.json();
    })
    .then(function(data) {
        // GPT가 생성한 답변 + 문서 참고 여부(배지) 표시
        addMessage("bot", data.answer, data.source, data.grounded);
    })
    .catch(function(error) {
        console.log(error)
        addMessage("bot", "서버 연결에 실패했습니다.");
    });
}

// 봇 답변에 "문서 참고 여부" 배지를 함께 표시 (F206)
// grounded 인자가 넘어온 /chat 응답에만 배지를 붙이고, 업로드/오류 안내 메시지에는 붙이지 않음
function addMessage(type, text, source, grounded) {
    const chatWindow = document.getElementById("chat-window");
    const messageDiv = document.createElement("div");
    messageDiv.className = "message " + type;
    messageDiv.innerText = text;

    if (type === "bot" && grounded !== undefined) {
        const badge = document.createElement("div");
        badge.className = "source-tag";
        if (grounded && source) {
            badge.innerText = "📄 약관 근거: " + source;
        } else {
            badge.innerText = "⚠️ 일반 AI 응답 (약관 미참고)";
        }
        messageDiv.appendChild(badge);
    }

    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}


document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});

// ============================================
// [추가] 업로드 패널 열기/닫기
// ============================================

function toggleUpload() {
    const panel = document.getElementById("upload-panel");
    // hidden 클래스가 있으면 제거, 없으면 추가
    panel.classList.toggle("hidden");
}


document.querySelector("#upload-panel button").addEventListener("click", async (e)=>{
    e.preventDefault();
    const fileInput = document.querySelector("#upload-panel input[type='file']");
    const file = fileInput.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try{
        const response = await fetch("http://localhost:8000/upload", {
            method: "POST",
            body: formData
        })
        const data = await response.json();
        console.log(data)
        addMessage("bot", data.message);
    }catch(err){
        addMessage("bot", "서버 연결에 실패했습니다.");
    }
})


// ============================================
// [추가] DB 초기화 비동기 함수
// ============================================
async function resetDatabase() {
    if (!confirm("정말로 데이터베이스를 초기화하시겠습니까?")) return;

    try {
        const response = await fetch("http://localhost:8000/reset-db", {
            method: "POST",
            headers: {
                "X-Requested-With": "XMLHttpRequest"
            }
        });
        const data = await response.json();
        if (data.success) {
            alert(data.message);
            // 전체 메시지 리셋 또는 봇 안내 메시지 출력
            const chatWindow = document.getElementById("chat-window");
            // chatWindow.innerHTML = ""; // 대화창 깨끗하게 정리
            addMessage("bot", "데이터베이스가 초기화되었습니다. 새 문서를 등록해 보세요!");
        } else {
            alert("초기화 실패: " + data.message);
        }
    } catch (error) {
        console.error(error);
        alert("서버 연결에 실패했습니다.");
    }
}
