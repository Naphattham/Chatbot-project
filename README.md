AutoXpert Chatbot 🚗💬
AI Chatbot สำหรับช่วยตอบข้อมูลเกี่ยวกับรถยนต์

📌 เกี่ยวกับโปรเจกต์
AutoXpert เป็น Chatbot ที่สามารถช่วยตอบคำถามเกี่ยวกับรถยนต์ เช่น:

ข้อมูลรุ่นรถยนต์และรายละเอียดของแต่ละรุ่น
ราคา, สมรรถนะ, อัตราเร่ง 0-100 กม./ชม.
ระบบขับเคลื่อน, ประเภทเครื่องยนต์ และคุณสมบัติอื่นๆ

🚀 วิธีการติดตั้งและใช้งาน
1️⃣ ติดตั้ง Dependencies
pip install -r requirements.txt

2️⃣ รันเซิร์ฟเวอร์ Flask
python app.py

3️⃣ เปิดเว็บแอปพลิเคชัน
เข้าใช้งานผ่านเบราว์เซอร์ที่:http://127.0.0.1:5000

📂 โครงสร้างโปรเจกต์
chatbot-project/
│── models/                # ไฟล์โมเดล AI (ถูก ignore ใน .gitignore)
│── static/                # ไฟล์ CSS, JavaScript และรูปภาพ
│── templates/             # ไฟล์ HTML สำหรับแสดงผลหน้าเว็บ
│── app.py                 # ไฟล์ Flask Backend
│── model.py               # ไฟล์ AI Model สำหรับตอบคำถาม
│── cars.json              # ข้อมูลรถยนต์ทั้งหมด
│── .gitignore             # กำหนดไฟล์ที่ไม่ต้องการ track ใน Git
│── README.md              # คำอธิบายโปรเจกต์ (ไฟล์นี้)

🎯 ฟีเจอร์หลัก
✅ รองรับการค้นหาข้อมูลของรถยนต์โดยใช้ AI Model
✅ แสดงข้อมูลราคา, สมรรถนะ, อัตราเร่ง และรายละเอียดอื่นๆ
✅ ใช้งานผ่าน เว็บเบราว์เซอร์ โดยมี UI ที่ใช้งานง่าย
✅ ไม่ต้องใช้ API ภายนอก ใช้ข้อมูลจาก ไฟล์ JSON ภายในโปรเจกต์

🛠 เทคโนโลยีที่ใช้
✅ Backend: Flask (Python)
✅ Frontend: HTML, CSS, JavaScript (Vanilla)
✅ AI Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
✅ Database: JSON (ไม่มีการเชื่อมต่อฐานข้อมูล)
✅ RAG (Retrieval-Augmented Generation): ใช้ข้อมูลจาก JSON และฝังข้อความ (Embeddings) เพื่อให้ LLM ประมวลผล
✅ LLM (Large Language Model): ใช้โมเดล Transformer เพื่อวิเคราะห์และสร้างข้อความตอบกลับแบบ AI

👨‍💻 ผู้พัฒนา
ชื่อ: Naphat Thammatheeroe (Neo)
GitHub: Naphattham
