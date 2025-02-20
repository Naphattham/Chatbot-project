from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import random
from sentence_transformers import SentenceTransformer, util
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_url_path='/static')
CORS(app)

# ✅ โหลดข้อมูลจาก JSON
with open("cars.json", "r", encoding="utf-8") as f:
    cars_data = json.load(f)

# ✅ โหลดโมเดลสำหรับวิเคราะห์ข้อความ (ใช้แบบรันในเครื่อง)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True, silent=True)

        if not data or "message" not in data:
            logging.warning("❌ ไม่พบ message ใน request")
            return jsonify({"error": "⚠️ กรุณาส่งข้อความ"}), 400

        user_input = data.get("message", "").strip()

        if not user_input:
            logging.warning("❌ ข้อความที่รับมาเป็นค่าว่าง")
            return jsonify({"response": "⚠️ กรุณาป้อนข้อความ"}), 400

        logging.info(f"📩 รับข้อความจากผู้ใช้: {user_input}")

        # ✅ ค้นหาข้อมูลที่เกี่ยวข้อง
        relevant_car = search_car_data(user_input)

        # ✅ สร้างคำตอบจากข้อมูลที่เจอ
        response = generate_text_response(user_input, relevant_car)

        logging.info(f"📤 ตอบกลับผู้ใช้: {response}")

        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"❌ เกิดข้อผิดพลาด: {str(e)}", exc_info=True)
        return jsonify({"error": "⚠️ เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์"}), 500

def search_car_data(query):
    """ค้นหาข้อมูลรถยนต์ที่เกี่ยวข้อง"""
    query_embedding = model.encode(query, convert_to_tensor=True)
    car_texts = [car["รุ่น"] + " " + " ".join(car.get("aliases", [])) for car in cars_data]
    car_embeddings = model.encode(car_texts, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(query_embedding, car_embeddings)[0]
    best_match_idx = scores.argmax().item()
    best_match_score = scores[best_match_idx].item()

    # ✅ ตั้งค่าคะแนนขั้นต่ำเพื่อกรองข้อมูลที่เกี่ยวข้อง
    if best_match_score < 0.6:
        return None

    return cars_data[best_match_idx]

def generate_text_response(user_input, car):
    """สร้างคำตอบจากข้อมูลที่ดึงมา"""
    if not car:
        return "⚠️ ไม่พบข้อมูลเกี่ยวกับรถรุ่นนี้ กรุณาลองสอบถามรุ่นอื่น"

    # ✅ ดึงข้อมูลที่เกี่ยวข้อง
    model_name = car["รุ่น"]
    year = car.get("ปี", "ไม่ระบุ")
    acceleration = car["สมรรถนะ"].get("อัตราเร่ง_0_100_กม_ชม", "ไม่ระบุ")
    top_speed = car["สมรรถนะ"].get("ความเร็วสูงสุด", "ไม่ระบุ")
    drive_system = car.get("ระบบขับเคลื่อน", "ไม่ระบุ")
    price_info = "\n".join([f"- {k}: {v}" for k, v in car["ราคา"].items()])

    # ✅ กำหนดรูปแบบการตอบให้มีความเป็นธรรมชาติ
    response_templates = [
        f"🚗 {model_name} ปี {year} มีสมรรถนะที่ดี โดยสามารถเร่ง 0-100 กม./ชม. ได้ใน {acceleration} และความเร็วสูงสุด {top_speed} กม./ชม. ระบบขับเคลื่อน: {drive_system} 💰 ราคา:\n{price_info}",
        f"รุ่น {model_name} เป็นรถที่ได้รับความนิยมในปี {year} มีอัตราเร่ง 0-100 กม./ชม. อยู่ที่ {acceleration} และความเร็วสูงสุดที่ {top_speed} กม./ชม. ระบบขับเคลื่อนเป็นแบบ {drive_system} โดยมีราคาเริ่มต้นที่:\n{price_info}",
        f"ถ้าคุณกำลังมองหารถ {model_name} ปี {year} นี่คือรายละเอียดที่น่าสนใจ:\n- อัตราเร่ง 0-100 กม./ชม.: {acceleration}\n- ความเร็วสูงสุด: {top_speed}\n- ระบบขับเคลื่อน: {drive_system}\n- 💰 ราคา:\n{price_info}",
    ]

    return random.choice(response_templates)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
