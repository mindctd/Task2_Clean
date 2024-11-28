import pandas as pd
import joblib

# โหลดโมเดล
model_path = r'D:/streamlit/Task2_Clean/logistic_regression_model.pkl'
model = joblib.load(model_path)

# กำหนดคอลัมน์ที่โมเดลเรียนรู้
model_features = ['Lane Types_Ready', 'Lane Types_SENTRI', 'Lane Types_Standard',
                  'Crossing Types_Passenger', 'Crossing Types_Pedestrian',
                  'County_San Diego']

# โหลดข้อมูลใหม่
new_data_path = 'D:/streamlit/Task2_Clean/cleaned_dataset_final.csv'
new_data = pd.read_csv(new_data_path)

# เพิ่มคอลัมน์ใหม่สำหรับ Lane Types, Crossing Types, และ County
# Lane Types
new_data['Lane Types'] = new_data[['Lane Types_Ready', 'Lane Types_SENTRI', 'Lane Types_Standard']].idxmax(axis=1)
new_data['Lane Types'] = new_data['Lane Types'].str.replace('Lane Types_', '')

# Crossing Types
new_data['Crossing Types'] = new_data[['Crossing Types_Passenger', 'Crossing Types_Pedestrian']].idxmax(axis=1)
new_data['Crossing Types'] = new_data['Crossing Types'].str.replace('Crossing Types_', '')

# County
new_data['County'] = new_data[['County_San Diego']].idxmax(axis=1)
new_data['County'] = new_data['County'].str.replace('County_', '')

# ทำนายผลลัพธ์
new_data_encoded = new_data[model_features]  # ใช้เฉพาะฟีเจอร์ที่จำเป็น
predictions = model.predict(new_data_encoded)
prediction_probabilities = model.predict_proba(new_data_encoded)[:, 1]

# เพิ่มผลลัพธ์กลับไปใน DataFrame
new_data['Predicted_Delay'] = predictions
new_data['Probability_Delay'] = prediction_probabilities

# เลือกคอลัมน์สำหรับแสดงผล
result_data = new_data[['Lane Types', 'Crossing Types', 'County', 'Predicted_Delay', 'Probability_Delay']]

# แสดงผลลัพธ์
print(result_data)
