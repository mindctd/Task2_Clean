import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
# ตรวจสอบว่าที่อยู่ไฟล์ถูกต้อง
data = pd.read_csv("E:\JOB\Task2_Clean\cleaned_dataset.csv")

# คำนวณสถิติหลักสำหรับคอลัมน์ Delay
delay_stats = {
    "Mean": data["Delay"].mean(),  # ค่าเฉลี่ย
    "Median": data["Delay"].median(),  # ค่ามัธยฐาน
    "Mode": data["Delay"].mode().iloc[0],  # โหมด (ค่าที่เกิดซ้ำมากที่สุด)
    "Min": data["Delay"].min(),  # ค่าต่ำสุด
    "Max": data["Delay"].max(),  # ค่าสูงสุด
    "Range": data["Delay"].max() - data["Delay"].min(),  # ช่วง
    "Standard Deviation": data["Delay"].std(),  # ส่วนเบี่ยงเบนมาตรฐาน
    "25th Percentile": data["Delay"].quantile(0.25),  # ค่าที่เปอร์เซ็นไทล์ที่ 25
    "75th Percentile": data["Delay"].quantile(0.75)  # ค่าที่เปอร์เซ็นไทล์ที่ 75
}

# สร้าง DataFrame เพื่อแสดงผล
delay_stats_df = pd.DataFrame(delay_stats, index=["Delay"])
print(delay_stats_df)

# เลือกเฉพาะคอลัมน์ตัวเลข
numeric_data = data.select_dtypes(include=["number"])

# คำนวณสถิติหลัก
statistics = numeric_data.describe().T  # แสดงสถิติพื้นฐาน (Mean, Std, Min, 25%, Median, 75%, Max)

# เพิ่มคอลัมน์ Mode และ Range
statistics["Mode"] = numeric_data.mode().iloc[0]
statistics["Range"] = numeric_data.max() - numeric_data.min()

print(statistics)

stats = data.describe()
print(stats)

# วาดกราฟ
plt.figure(figsize=(8, 6))
sns.histplot(data["Delay"], kde=True, bins=20, color="blue")
plt.title("Distribution of Delay")
plt.xlabel("Delay (minutes)")
plt.ylabel("Frequency")
plt.show()
