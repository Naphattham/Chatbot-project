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

    if not car:
        return "⚠️ ไม่พบข้อมูลเกี่ยวกับรถรุ่นนี้ กรุณาลองสอบถามรุ่นอื่น"

    # ✅ ดึงข้อมูลที่เกี่ยวข้อง
    model_name = car["รุ่น"]
    year = car.get("ปี", "ไม่ระบุ")
    acceleration = car["สมรรถนะ"].get("อัตราเร่ง_0_100_กม_ชม", "ไม่ระบุ")
    top_speed = car["สมรรถนะ"].get("ความเร็วสูงสุด", "ไม่ระบุ")
    drive_system = car.get("ระบบขับเคลื่อน", "ไม่ระบุ")
    price_info = "\n".join([f"- {k}: {v}" for k, v in car["ราคา"].items()])

    # ✅ ตรวจสอบว่าผู้ใช้ถามเกี่ยวกับอะไร
    if "ราคา" in user_input:
        response_templates = [
            f"💰 {model_name} ปี {year} มีราคาดังนี้:\n{price_info}",
            f"📌 รุ่น {model_name} มีราคาขายที่:\n{price_info}",
            f"🛒 คุณกำลังสนใจ {model_name} อยู่ใช่ไหม? นี่คือราคาของมัน:\n{price_info}",
            f"💵 สำหรับ {model_name} ปี {year} นี่คือข้อมูลราคาล่าสุด:\n{price_info}",
            f"📊 นี่คือตารางราคาของ {model_name}:\n{price_info}",
            f"🔍 คุณถามมาพอดี! นี่คือราคา {model_name}:\n{price_info}",
            f"🛠️ เช็คราคาของ {model_name} ได้ที่นี่:\n{price_info}",
            f"🏷️ อยากรู้ราคาใช่ไหม? นี่เลย:\n{price_info}",
            f"✅ ราคาของ {model_name} ปี {year} เป็นดังนี้:\n{price_info}",
            f"🧐 สำหรับราคาของ {model_name} ดูได้ที่นี่เลย:\n{price_info}"
        ]
        return random.choice(response_templates)

    if "อัตราเร่ง" in user_input or "0-100" in user_input:
        response_templates = [
            f"🚀 {model_name} ทำอัตราเร่ง 0-100 กม./ชม. ได้ใน {acceleration}",
            f"🏎️ อัตราเร่งของ {model_name} อยู่ที่ {acceleration}",
            f"⚡ {model_name} สามารถเร่ง 0-100 กม./ชม. ภายใน {acceleration}",
            f"🚦 ถ้าคุณอยากรู้เรื่องความเร็ว {model_name} ทำได้ใน {acceleration}",
            f"💨 ความเร็วต้นของ {model_name} อยู่ที่ {acceleration}",
            f"🏁 สนใจเรื่องอัตราเร่งใช่ไหม? {model_name} ทำได้ {acceleration}",
            f"📢 ข่าวดี! {model_name} สามารถเร่ง 0-100 กม./ชม. ได้ที่ {acceleration}",
            f"🕐 {model_name} สามารถเร่งจาก 0-100 กม./ชม. ได้ที่ {acceleration}",
            f"🔰 อัตราเร่งของ {model_name} คือ {acceleration}",
            f"🔥 คุณจะรู้สึกถึงพลังของ {model_name} ที่ {acceleration}"
        ]
        return random.choice(response_templates)

    if "ความเร็วสูงสุด" in user_input:
        response_templates = [
            f"🏎️ {model_name} ทำความเร็วสูงสุดได้ {top_speed} กม./ชม.",
            f"🚀 ความเร็วสูงสุดของ {model_name} คือ {top_speed} กม./ชม.",
            f"📢 คุณถามเรื่องความเร็วใช่ไหม? {model_name} ไปได้สูงสุดที่ {top_speed} กม./ชม.",
            f"🔥 {model_name} ถูกออกแบบให้ทำความเร็วได้ถึง {top_speed} กม./ชม.",
            f"⚡ ด้วยเทคโนโลยีล่าสุด {model_name} ไปได้ถึง {top_speed} กม./ชม.",
            f"🏁 คุณจะสัมผัสความเร็วที่ {top_speed} กม./ชม. กับ {model_name}",
            f"🔰 {model_name} วิ่งเร็วสุดที่ {top_speed} กม./ชม.",
            f"🎯 ข้อมูลล่าสุดบอกว่า {model_name} มีความเร็วสูงสุดที่ {top_speed} กม./ชม.",
            f"🧐 ความเร็วสูงสุดของ {model_name} คือ {top_speed} กม./ชม.",
            f"🚦 ขับขี่อย่างมั่นใจ {model_name} วิ่งสูงสุดที่ {top_speed} กม./ชม."
        ]
        return random.choice(response_templates)

    if "ระบบขับเคลื่อน" in user_input:
        response_templates = [
            f"🛞 ระบบขับเคลื่อนของ {model_name} คือ {drive_system}",
            f"⚙️ {model_name} ใช้ระบบขับเคลื่อนแบบ {drive_system}",
            f"🔍 คุณกำลังสนใจระบบขับเคลื่อนของ {model_name} ใช่ไหม? มันเป็นแบบ {drive_system}",
            f"🔧 {model_name} ขับเคลื่อนด้วยระบบ {drive_system}",
            f"📢 ระบบขับเคลื่อนของ {model_name} ออกแบบมาเป็น {drive_system}",
            f"🏎️ หากคุณอยากรู้เกี่ยวกับการขับเคลื่อน {model_name} ใช้ {drive_system}",
            f"🛠️ ระบบขับเคลื่อนของ {model_name} คือ {drive_system}",
            f"📌 {model_name} มาพร้อมกับระบบขับเคลื่อน {drive_system}",
            f"🧐 ระบบขับเคลื่อนที่ใช้ใน {model_name} คือ {drive_system}",
            f"🚗 {model_name} ใช้ระบบ {drive_system} เพื่อให้การขับขี่ดีที่สุด"
        ]
        return random.choice(response_templates)

    # ✅ ถ้าผู้ใช้ไม่ได้ถามเฉพาะเจาะจง ตอบข้อมูลทั่วไป
    response_templates = [
        f"🚗 {model_name} ปี {year} มีอัตราเร่ง 0-100 กม./ชม. ที่ {acceleration} และความเร็วสูงสุด {top_speed} กม./ชม. ระบบขับเคลื่อน: {drive_system} 💰 ราคา:\n{price_info}",
        f"📌 {model_name} เป็นรถที่ได้รับความนิยมในปี {year} มีอัตราเร่ง 0-100 กม./ชม. ที่ {acceleration} และความเร็วสูงสุด {top_speed} กม./ชม.",
        f"🛠️ ถ้าคุณกำลังมองหารถ {model_name} ปี {year} นี่คือรายละเอียดที่น่าสนใจ:\n- อัตราเร่ง 0-100 กม./ชม.: {acceleration}\n- ความเร็วสูงสุด: {top_speed}\n- ระบบขับเคลื่อน: {drive_system}\n- 💰 ราคา:\n{price_info}",
        f"🏁 ข้อมูลของ {model_name} ปี {year}:\n- อัตราเร่ง: {acceleration}\n- ความเร็วสูงสุด: {top_speed}\n- ระบบขับเคลื่อน: {drive_system}\n💰 ราคา:\n{price_info}",
        f"🧐 {model_name} ปี {year} มาพร้อมกับสมรรถนะที่ยอดเยี่ยม! อัตราเร่ง {acceleration}, ความเร็วสูงสุด {top_speed} กม./ชม.",
        f"🎯 รายละเอียดของ {model_name} ปี {year}: ระบบขับเคลื่อน {drive_system}, อัตราเร่ง {acceleration}, ความเร็วสูงสุด {top_speed} กม./ชม.",
        f"📢 สนใจ {model_name} ใช่ไหม? นี่คือรายละเอียด:\n- ความเร็วสูงสุด {top_speed} กม./ชม.\n- ระบบขับเคลื่อน {drive_system}\n💰 ราคา:\n{price_info}",
        f"🔧 {model_name} ปี {year} เป็นรถที่มีสมรรถนะดีเยี่ยม! ราคาเริ่มต้นที่:\n{price_info}",
        f"🛒 สนใจซื้อ {model_name} ปี {year} ไหม? นี่คือข้อมูลของมัน:\n- อัตราเร่ง {acceleration}\n- ระบบขับเคลื่อน {drive_system}",
        f"📌 {model_name} ปี {year} มีราคาที่แข่งขันได้ โดยมีข้อมูลสมรรถนะดังนี้:\n- ความเร็วสูงสุด {top_speed}\n- ระบบขับเคลื่อน {drive_system}"
    ]
    
    return random.choice(response_templates)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
