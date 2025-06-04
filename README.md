
# 🐾 AUC System

เว็บแอปพลิเคชันที่ช่วยจัดลำดับความรุนแรงของโรคที่เกี่ยวข้องกับระบบทางเดินปัสสาวะในสัตว์ โดยใช้ระบบจัดเก็บข้อมูลและประมวลผลผ่าน API และฐานข้อมูล พร้อมอินเทอร์เฟซผู้ใช้ที่ใช้งานง่าย เหมาะสำหรับสัตวแพทย์ นักวิจัย และนักศึกษาสัตวแพทย์

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
- **Elysia.js**: Web framework ที่ประสิทธิภาพสูง
- **Mongoose**: เชื่อมต่อ MongoDB
- **jsonwebtoken**, **bcryptjs**: สำหรับ auth system
- **Flask + XGBoost**: ให้บริการ REST API และวิเคราะห์ความรุนแรงของโรค

### Frontend
- **Next.js**: React Framework สำหรับ SSR
- **MUI (Material UI)**: สำหรับ UI ที่ใช้งานง่าย
- **TailwindCSS**: สำหรับ utility-first styling
- **JWT-decode**, **UUID**: สำหรับจัดการ token และ unique ID

---

## 🗂️ โครงสร้างโปรเจกต์

### 📁 backend/app

```
src/
├── middleware/         # ตรวจสอบ token
│   └── VerifyToken.ts
├── models/             # โมเดลข้อมูล Mongoose
│   ├── BlacklistedToken.ts
│   ├── Patient.ts
│   └── User.ts
├── routes/             # API routes
│   ├── auth.ts
│   ├── patient.ts
│   └── users.ts
├── types/              # Type definitions (TS)
│   ├── AuthBoody.ts
│   ├── PatientBody.ts
│   └── UserBody.ts
├── index.ts            # Entry point ของ backend
```

- `predictor.py`: Flask API สำหรับ XGBoost
- `requirements.txt`: รายการ Python packages
- `tsconfig.json`, `package.json`: ตั้งค่าโปรเจกต์และ dependency

---

### 📁 client/frontend

```
app/
├── components/         # ส่วนประกอบของ UI
│   ├── AddPatientForm.tsx
│   ├── EditPatientForm.tsx
│   ├── GeneralInfoCard.tsx
│   ├── LoginPage.tsx
│   ├── PatientCard.tsx
│   ├── Sidebar.tsx
│   └── UrinalysisInfoCard.tsx
├── lib/
│   └── fetchWithToken.ts
├── login/              # หน้าล็อกอิน
│   └── page.tsx
├── patients/           # หน้าแสดงและจัดการผู้ป่วย
│   ├── [id]/edit/page.tsx
│   ├── [id]/page.tsx
│   └── new/page.tsx
├── types/              # TS types
│   ├── Auth.ts
│   ├── Patient.ts
│   └── Urinalysis.ts
├── layout.tsx, page.tsx, globals.css
```

---

## 🚀 วิธีเริ่มต้นใช้งาน

### 🔧 การติดตั้ง 

แตกไฟล์ zip ออกมาแล้วเปิดด้วย VS Code

#### ✅ ติดตั้ง backend
```bash
cd backend/app
npm install
npm run dev
```

#### ✅ ติดตั้ง frontend
```bash
cd client/frontend
npm install
npm run dev
```

#### ✅ ติดตั้ง Python server
> จำเป็นต้องมี Python 3.8+ และ `pip` ติดตั้งไว้แล้ว
```bash
cd backend
pip install -r requirements.txt
python predictor.py
```

---

## 🧪 วิธีใช้งาน

1. เข้าสู่ระบบด้วยบัญชีที่มีอยู่ (หรือสมัครใช้งาน ผ่านการยิงapi)
2. เพิ่มข้อมูลผู้ป่วยผ่านหน้า "Add Patient"
3. ระบบจะส่งข้อมูลไปยัง backend และ Flask API เพื่อวิเคราะห์
4. หน้า UI จะแสดงผลระดับความรุนแรง (เช่น Moderate, Severe ฯลฯ)
5. สามารถแก้ไข ลบ และดูประวัติย้อนหลังได้
6. รองรับ token-based authentication (JWT) สำหรับความปลอดภัย


