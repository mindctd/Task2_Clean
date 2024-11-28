import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ตรวจสอบพาธของไฟล์ Python ปัจจุบัน
current_file_path = os.path.abspath(__file__)  # พาธของไฟล์ Python ปัจจุบัน
current_folder_path = os.path.dirname(current_file_path)  # โฟลเดอร์ที่ไฟล์อยู่

print("พาธของไฟล์:", current_file_path)
print("โฟลเดอร์ที่ไฟล์อยู่:", current_folder_path)

# ตัวอย่างการบันทึกไฟล์ CSV ในโฟลเดอร์เดียวกับไฟล์ Python
data = {'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']}
df = pd.DataFrame(data)

output_file_path = os.path.join(current_folder_path, 'output_file.csv')
df.to_csv(output_file_path, index=False)

print(f"บันทึกไฟล์ CSV ที่: {output_file_path}")

# โหลด Dataset
df = pd.read_csv('D:/pandas_task2/cleaned_dataset.csv')

# ตรวจสอบข้อมูลเบื้องต้น
print("ข้อมูลเบื้องต้น:")
print(df.info())

# 1. แปลงคอลัมน์วันที่ (Date) ให้เป็นตัวเลข (แปลงแต่จะตัดออกใน X)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # แปลงเป็น datetime
    df['Date'] = (df['Date'] - pd.Timestamp("1970-01-01")).dt.days  # แปลงเป็นจำนวนวัน

# 2. กำหนดตัวแปรต้นและตัวแปรตาม
X = df[['Lane Types', 'Crossing Types', 'Date', 'County']]  # ตัวแปรต้น
y = (df['Delay'] > 0.5).astype(int)  # แปลง Delay ให้เป็น Binary (0 หรือ 1)

# 3. เข้ารหัสตัวแปรหมวดหมู่ด้วย One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# 4. ตัดคอลัมน์ Date ออก
X = X.drop(columns=['Date'], errors='ignore')

# 5. แปลงข้อมูลใน X ให้เป็น float
X = X.astype(float)

# ตรวจสอบชนิดข้อมูล (Data Types) ของตัวแปรต้น
print("ชนิดข้อมูลของตัวแปรต้นหลังตัด Date ออก:")
print(X.dtypes)

# คำนวณ Correlation ระหว่างตัวแปรต้นและตัวแปรตาม
df_combined = pd.concat([X, y.rename("Delay")], axis=1)
correlation_matrix = df_combined.corr()
print("Correlation ระหว่างตัวแปรต้นและ Delay:")
print(correlation_matrix['Delay'].sort_values(ascending=False))

# 6. แบ่งข้อมูลเป็นชุด Training และ Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. ใช้ SMOTE เพื่อปรับสมดุลข้อมูล
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 8. สร้างและฝึกโมเดล Logistic Regression
model = LogisticRegression()
model.fit(X_train_balanced, y_train_balanced)

# บันทึกโมเดลที่ฝึกไว้
model_path = r'D:/streamlit/Task2_Clean/logistic_regression_model.pkl'  # กำหนดพาธของไฟล์ที่ต้องการบันทึก
joblib.dump(model, model_path)
print(f"โมเดลถูกบันทึกเรียบร้อยแล้วที่: {model_path}")

print("ไฟล์โมเดลมีอยู่หรือไม่:", os.path.exists(model_path))

# 9. ทำนายผลบนชุด Testing
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # ความน่าจะเป็นของคลาส "Delay" (1)

# 10. คำนวณตัวชี้วัด (Evaluation Metrics)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# แสดงผลตัวชี้วัด
print(f"Accuracy (Overall): {accuracy:.2f}")
print(f"Precision (Delay): {precision:.2f}")
print(f"Recall (Delay): {recall:.2f}")
print(f"F1-Score (Delay): {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")

# รายงานตัวชี้วัดทั้งหมด
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Delay", "Delay"]))

# วาดกราฟ ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # เส้นฐาน
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# วาดกราฟ Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# รวมตัวแปรต้น (X) และตัวแปรตาม (y) ลงใน DataFrame
cleaned_data = pd.concat([X, y.rename("Delay")], axis=1)

# แก้ไขพาธที่ต้องการบันทึกไฟล์ CSV (พาธที่คุณต้องการ)
output_file_path = r'D:/streamlit/Task2_Clean/cleaned_dataset_final.csv'  # แก้ไขพาธนี้

# บันทึกไฟล์ CSV
cleaned_data.to_csv(output_file_path, index=False)
print(f"บันทึกข้อมูลที่ปรับปรุงแล้วเรียบร้อยที่: {output_file_path}")
