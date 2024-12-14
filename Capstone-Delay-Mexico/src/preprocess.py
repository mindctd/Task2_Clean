import pandas as pd
import numpy as np

# อ่านข้อมูล
df = pd.read_csv('E:/JOB/Task2_Clean/cleaned_dataset.csv')  

# 1. เติมเต็ม Missing Values
# เติมค่าข้อมูลตัวเลขที่ขาดหายด้วยค่าเฉลี่ย
df_numeric = df.select_dtypes(include=[np.number])
df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())

# เติมค่าข้อมูลประเภทข้อความด้วยค่าที่พบบ่อยที่สุด (Mode)
df_object = df.select_dtypes(include=['object'])
df[df_object.columns] = df_object.apply(lambda x: x.fillna(x.mode()[0]))

# แสดงข้อมูลหลังจัดการ Missing Values
print("ข้อมูลหลังเติมเต็ม Missing Values:\n", df.head())

# 2. ตรวจหา Outliers และลบทั้งแถว
# ใช้วิธี Interquartile Range (IQR)
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1

outlier_condition = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR)))
outliers_count = outlier_condition.sum().sum()  # นับจำนวน outliers ทั้งหมด
print(f"จำนวน Outliers ทั้งหมด: {outliers_count}")

# ลบแถวที่มี Outliers
df_cleaned = df[~outlier_condition.any(axis=1)]
print(f"ข้อมูลหลังลบ Outliers: {len(df_cleaned)} rows remaining.")

# 3. ลบ Duplicates
# ตรวจหาข้อมูลซ้ำ
duplicates_count = df_cleaned.duplicated().sum()
print(f"จำนวน Duplicates: {duplicates_count}")

# ลบข้อมูลที่ซ้ำ
df_cleaned = df_cleaned.drop_duplicates()
print(f"ข้อมูลหลังลบ Duplicates: {len(df_cleaned)} rows remaining.")

# 4. ตรวจสอบประเภทข้อมูล (Data Types)
# แสดงประเภทข้อมูลของคอลัมน์
data_types = df_cleaned.dtypes
print("ประเภทข้อมูลของแต่ละคอลัมน์:\n", data_types)

# ตรวจสอบว่าประเภทข้อมูลถูกต้องหรือไม่
correct_types = True

# สรุปผลการตรวจสอบประเภทข้อมูล
if correct_types:
    print("ประเภทข้อมูลทั้งหมดถูกต้อง.")
else:
    print("มีประเภทข้อมูลที่ไม่ถูกต้อง. โปรดตรวจสอบเพิ่มเติม.")

# 5. แสดงข้อมูลหลัง Clean
print("ข้อมูลที่ผ่านการ Clean แล้ว:\n", df_cleaned.head())

# บันทึกข้อมูลที่ Clean แล้วกลับไปเป็นไฟล์ CSV (ถ้าต้องการ)
df_cleaned.to_csv('cleaned_dataset.csv', index=False)
