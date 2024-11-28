import pandas as pd
import matplotlib.pyplot as plt

# 1. โหลดข้อมูลจากไฟล์ CSV
input_file_path = 'D:/streamlit/Task2_Clean/cleaned_dataset_final.csv'  # ระบุพาธไฟล์ CSV
df = pd.read_csv(input_file_path)

# คำนวณจำนวนความล่าช้าที่เกิดขึ้นในแต่ละประเภทเลน (Lane Types)
lane_types_columns = ['Lane Types_Ready', 'Lane Types_SENTRI', 'Lane Types_Standard']
lane_delay_count = df[df['Delay'] == 1][lane_types_columns].sum().sort_values(ascending=False)

# คำนวณจำนวนความล่าช้าที่เกิดขึ้นในแต่ละประเภทการข้าม (Crossing Types)
crossing_types_columns = ['Crossing Types_Passenger', 'Crossing Types_Pedestrian']
crossing_delay_count = df[df['Delay'] == 1][crossing_types_columns].sum().sort_values(ascending=False)

# คำนวณจำนวนความล่าช้าที่เกิดขึ้นในแต่ละจังหวัด (County)
county_columns = ['County_San Diego']  # คุณสามารถเพิ่มคอลัมน์อื่นๆ ที่เกี่ยวข้องในนี้
county_delay_count = df[df['Delay'] == 1][county_columns].sum().sort_values(ascending=False)

# แสดงผลลัพธ์
print("\nจำนวนความล่าช้าที่เกิดขึ้นในแต่ละประเภทเลน:")
print(lane_delay_count)

print("\nจำนวนความล่าช้าที่เกิดขึ้นในแต่ละประเภทการข้าม:")
print(crossing_delay_count)

print("\nจำนวนความล่าช้าที่เกิดขึ้นในแต่ละจังหวัด:")
print(county_delay_count)

# การแสดงกราฟ
plt.figure(figsize=(10, 6))

# กราฟแสดงจำนวนความล่าช้าที่เกิดขึ้นในแต่ละประเภทเลน
plt.subplot(1, 3, 1)
lane_delay_count.plot(kind='bar', color='skyblue')
plt.title('จำนวนความล่าช้าที่เกิดขึ้นในแต่ละประเภทเลน')
plt.xlabel('ประเภทเลน')
plt.ylabel('จำนวนความล่าช้า')
plt.xticks(rotation=45)

# กราฟแสดงจำนวนความล่าช้าที่เกิดขึ้นในแต่ละประเภทการข้าม
plt.subplot(1, 3, 2)
crossing_delay_count.plot(kind='bar', color='lightgreen')
plt.title('จำนวนความล่าช้าที่เกิดขึ้นในแต่ละประเภทการข้าม')
plt.xlabel('ประเภทการข้าม')
plt.ylabel('จำนวนความล่าช้า')
plt.xticks(rotation=45)

# กราฟแสดงจำนวนความล่าช้าที่เกิดขึ้นในแต่ละจังหวัด
plt.subplot(1, 3, 3)
county_delay_count.plot(kind='bar', color='lightcoral')
plt.title('จำนวนความล่าช้าที่เกิดขึ้นในแต่ละจังหวัด')
plt.xlabel('จังหวัด')
plt.ylabel('จำนวนความล่าช้า')
plt.xticks(rotation=45)

# แสดงกราฟ
plt.tight_layout()
plt.show()