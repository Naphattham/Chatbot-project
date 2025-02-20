document.addEventListener("DOMContentLoaded", function () {
    console.log("✅ JavaScript Loaded!");

    const chatBox = document.getElementById("chat-box");
    const inputField = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");

    if (!inputField || !sendButton) {
        console.error("❌ ไม่พบ input หรือปุ่มส่งข้อความ");
        return;
    }

    console.log("✅ พบ input และปุ่มส่งข้อความ");

    // ✅ แสดงข้อความต้อนรับทันทีที่หน้าโหลด
    appendMessage("bot", "🚗 สวัสดีครับ! สามารถสอบถามข้อมูลเกี่ยวกับรถยนต์ได้เลย!");

    sendButton.addEventListener("click", sendMessage);
    inputField.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });

    function sendMessage() {
        const userMessage = inputField.value.trim();
        if (userMessage === "") return;

        console.log("📨 กำลังส่งข้อความ:", userMessage);

        appendMessage("user", userMessage);
        inputField.value = "";

        fetch("http://127.0.0.1:5000/chat", {  // ✅ เปลี่ยน URL ให้ถูกต้อง
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: userMessage })
        })
        .then(response => response.json())
        .then(data => {
            if (data.response) {
                appendMessage("bot", data.response);
            } else {
                appendMessage("bot", "⚠️ ไม่พบข้อมูลที่เกี่ยวข้อง กรุณาลองใหม่");
            }
        })
        .catch(error => {
            console.error("❌ เกิดข้อผิดพลาด:", error);
            appendMessage("bot", "⚠️ มีข้อผิดพลาดในการเชื่อมต่อกับ API");
        });
    }

    function appendMessage(sender, message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("chat-message", sender);
        messageElement.innerHTML = `<p>${message.replace(/\n/g, "<br>")}</p>`;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
