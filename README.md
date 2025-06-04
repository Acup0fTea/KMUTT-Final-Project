# 🐾  AUC system

เว็บแอปพลิเคชันที่ช่วยจัดลำดับความรุนแรงของโรคที่เกี่ยวข้องกับ ระบบทางเดินปัสสาวะในสัตว์ โดยใช้ระบบจัดเก็บข้อมูลและประมวลผลผ่าน API และฐานข้อมูล พร้อมอินเทอร์เฟซผู้ใช้ที่ใช้งานง่าย เหมาะสำหรับสัตวแพทย์ นักวิจัย และนักศึกษาสัตวแพทย์

---

## 🧠 ฟีเจอร์หลัก

- วิเคราะห์และจัดลำดับความร้ายแรงของโรคในระบบทางเดินปัสสาวะของสัตว์
- จัดเก็บข้อมูลประวัติผู้ป่วยและการวินิจฉัย
- ระบบยืนยันตัวตนด้วย JWT
- รองรับ Redis caching และ MongoDB
- UI ด้วย React + Next.js + Material UI

---

## 🏗️ โครงสร้างระบบ

### Backend
- Elysia.js: Web framework ที่ประสิทธิภาพสูง
- Mongoose: เชื่อมต่อ MongoDB
- jsonwebtoken และ bcryptjs: สำหรับ auth system
- Flask: สำหรับให้บริการ REST API
- XGBoost: ใช้ในการจัดอันดับความรุนแรง

### Frontend
- Next.js: React Framework สำหรับ SSR
- MUI (Material UI): สำหรับ UI ที่ใช้งานง่าย
- TailwindCSS: สำหรับ utility-first styling
- JWT-decode, UUID: สำหรับจัดการ token และ unique ID

---

## 🚀 วิธีเริ่มต้นใช้งาน

### 🔧 การติดตั้ง 
แตกไฟล์ zip ออกมาเเล้วทำการเปิดด้วย vscode

# ติดตั้ง backend
cd เข้าไปที่directory app เเละ npm install

# เริ่มรันเซิร์ฟเวอร์
npm run dev

# ติดตั้ง frontend
npm install

# เริ่มต้นclient server
npm run dev

# ติดตั้ง python server
> จำเป็นต้องมี Python 3.8+ และ `pip` ติดตั้งไว้แล้ว
pip install -r requirements.txt

# เริ่มต้น python server
python predictor.py
