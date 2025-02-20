from sentence_transformers import SentenceTransformer, util
import json
import os

# 🔹 โหลดโมเดล Sentence Transformer
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name, cache_folder="./models")

current_dir = os.path.dirname(__file__)  # หาตำแหน่งไฟล์ model.py
cars_file_path = os.path.join(current_dir, "cars.json")  # กำหนด path

with open(cars_file_path, "r", encoding="utf-8") as f:
    docs = json.load(f)


if isinstance(docs, dict):
    docs = [docs]  

# 🔹 ฟังก์ชันแปลงข้อมูลรถเป็นข้อความ
def format_car_info(car):
    try:
        สมรรถนะ = car.get("สมรรถนะ", {})
        ราคา = car.get("ราคา", {})
        price_info = "\n".join([f"- {key}: {value}" for key, value in ราคา.items() if value])

        return f"🚗 ยี่ห้อ: {car.get('ยี่ห้อ', 'ไม่ระบุ')}, รุ่น: {car.get('รุ่น', 'ไม่ระบุ')}, ปี: {car.get('ปี', 'ไม่ระบุ')}, \
อัตราเร่ง 0-100 กม./ชม.: {สมรรถนะ.get('อัตราเร่ง_0_100_กม_ชม', 'ไม่ระบุ')}, \
ความเร็วสูงสุด: {สมรรถนะ.get('ความเร็วสูงสุด', 'ไม่ระบุ')}, \
💰 ราคา:\n{price_info if price_info else 'ไม่มีข้อมูล'}, \
ระบบขับเคลื่อน: {car.get('ระบบขับเคลื่อน', 'ไม่ระบุ')}"
    except Exception as e:
        return f"⚠️ ข้อมูลรถไม่ถูกต้อง: {str(e)}"

# 🔹 ฟังก์ชันค้นหาข้อมูล
def retrieve_response(query):
    """ค้นหาข้อมูลรถยนต์ที่ใกล้เคียงที่สุดกับ Query"""
    query = query.lower().strip()

    # 🔹 ตรวจสอบว่าผู้ใช้ถามถึงยี่ห้อไหน
    brand_names = set(car["ยี่ห้อ"].lower() for car in docs if "ยี่ห้อ" in car)
    detected_brand = next((brand for brand in brand_names if brand in query), None)

    if detected_brand:
        return list_brand_models(detected_brand)

    for car in docs:
        model_name = car["รุ่น"].lower()
        if all(word in query for word in model_name.split()):
            return process_query(query, car)


    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, model.encode([format_car_info(car) for car in docs], convert_to_tensor=True))[0]
    
    best_match_idx = scores.argmax().item()
    best_match_score = scores[best_match_idx].item()

    similarity_threshold = 0.7  
    if best_match_score < similarity_threshold:
        return "⚠️ ไม่มีอยู่ในข้อมูล กรุณาถามเกี่ยวกับรถยนต์ที่อยู่ในระบบ"

    return process_query(query, docs[best_match_idx])

# 🔹 ฟังก์ชันแสดงรุ่นทั้งหมดของยี่ห้อที่ถูกถาม
def list_brand_models(brand):
    """คืนค่ารายการรุ่นทั้งหมดของแบรนด์ที่ถูกถาม"""
    models = [car["รุ่น"] for car in docs if car.get("ยี่ห้อ", "").lower() == brand]
    
    if models:
        return f"🚗 ยี่ห้อ {brand.capitalize()} มีทั้งหมด {len(models)} รุ่น ได้แก่:\n- " + "\n- ".join(models)
    return f"⚠️ ไม่มีข้อมูลเกี่ยวกับยี่ห้อ {brand.capitalize()} ในระบบ"

# 🔹 ฟังก์ชันจัดการการตอบกลับ
def process_query(query, car):
    """จัดการกับคำถามของผู้ใช้ และเลือกตอบเฉพาะสิ่งที่เกี่ยวข้อง"""
    สมรรถนะ = car.get("สมรรถนะ", {})
    ราคา = car.get("ราคา", {})

    if "ราคา" in query:
        price_info = "\n".join([f"- {key}: {value}" for key, value in ราคา.items() if value])
        return f"💰 รุ่น {car.get('รุ่น', 'ไม่ระบุ')} มีราคาดังนี้:\n{price_info if price_info else 'ไม่มีข้อมูล'}"

    if "0-100" in query or "อัตราเร่ง" in query:
        return f"🚀 อัตราเร่ง 0-100 กม./ชม. ของ {car.get('รุ่น', 'ไม่ระบุ')} คือ {สมรรถนะ.get('อัตราเร่ง_0_100_กม_ชม', 'ไม่มีข้อมูล')}"

    if "ปี" in query or "ผลิตปี" in query:
        return f"{car.get('รุ่น', 'ไม่ระบุ')} ผลิตในปี {car.get('ปี', 'ไม่มีข้อมูล')}"

    if "ความเร็วสูงสุด" in query:
        return f"🚀 ความเร็วสูงสุดของ {car.get('รุ่น', 'ไม่ระบุ')} คือ {สมรรถนะ.get('ความเร็วสูงสุด', 'ไม่มีข้อมูล')}"

    if "ระบบขับเคลื่อน" in query:
        return f"🛞 ระบบขับเคลื่อนของ {car.get('รุ่น', 'ไม่ระบุ')} คือ {car.get('ระบบขับเคลื่อน', 'ไม่มีข้อมูล')}"
    
    return format_car_info(car)
