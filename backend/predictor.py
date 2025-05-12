from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__)

# โหลดโมเดล XGBoost
booster = xgb.Booster()
booster.load_model("xgboost_model.json")

# รายการฟีเจอร์ที่โมเดลต้องการ
FEATURE_ORDER = [
    "Color", "Appearance", "Specific Gravity", "pH", "Protein", "Glucose",
    "Ketone", "Bilirubin", "Blood", "WBC", "RBC", "Casts",
    "Amorphous Phosphate Crystals", "Bacteria", "Squamous Epithelia", "Oval Fat Body"
]

# Mapping ค่าที่เป็น string ให้เป็น integer ตามที่โมเดลฝึกไว้
VALUE_MAPPING = {
    "Color": {"yellow": 0, "amber": 1, "red": 2},
    "Appearance": {"clear": 0, "cloudy": 1},
    "Protein": {"negative": 0, "trace": 1, "1+": 2, "2+": 3, "3+": 4, "4+": 5},
    "Glucose": {"negative": 0, "trace": 1, "1+": 2, "2+": 3, "3+": 4},
    "Ketone": {"negative": 0, "trace": 1, "1+": 2},
    "Bilirubin": {"negative": 0, "trace": 1, "1+": 2},
    "Blood": {"negative": 0, "trace": 1, "1+": 2, "2+": 3, "3+": 4},
    "WBC": {"none": 0, "few": 1, "moderate": 2, "many": 3},
    "RBC": {"none": 0, "few": 1, "moderate": 2, "many": 3},
    "Casts": {"none": 0, "hyaline": 1, "granular": 2},
    "Amorphous Phosphate Crystals": {"none": 0, "few": 1, "moderate": 2},
    "Bacteria": {"none": 0, "few": 1, "many": 2},
    "Squamous Epithelia": {"none": 0, "few": 1},
    "Oval Fat Body": {"none": 0, "few": 1},
}

# Preprocess ข้อมูลจาก JSON ให้กลายเป็น DataFrame ที่เหมาะกับ XGBoost
def preprocess_input(data):
    processed = {}
    for feature in FEATURE_ORDER:
        raw_value = data.get(feature, 0)
        if feature in VALUE_MAPPING:
            if isinstance(raw_value, str):
                key = raw_value.lower()
            else:
                key = str(raw_value).lower()
            processed[feature] = VALUE_MAPPING[feature].get(key, 0)
        else:
            try:
                processed[feature] = float(raw_value)
            except ValueError:
                processed[feature] = 0.0
    return pd.DataFrame([processed])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_json = request.json
        print("📥 Received JSON:", input_json)

        df = preprocess_input(input_json)
        print("📊 Processed DataFrame:\n", df)

        dmatrix = xgb.DMatrix(df[FEATURE_ORDER])
        prediction = booster.predict(dmatrix)
        print("🔮 Raw prediction:", prediction)

        # ✅ ใช้ argmax แทนการ round
        class_index = int(np.argmax(prediction[0]))
        severity_map = {0: "Low", 1: "Moderate", 2: "High"}
        severity = severity_map.get(class_index, "Low")

        return jsonify({"severityLevel": severity})

    except Exception as e:
        print("❌ Error occurred during prediction:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
