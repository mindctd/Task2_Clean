import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# โหลดข้อมูล
data = pd.read_csv("D:\pandas_task2\cleaned_dataset.csv")

# ฟังก์ชันสำหรับคำนวณสถิติหลัก
def calculate_statistics(column):
    stats = {
        "Mean": column.mean(),
        "Median": column.median(),
        "Mode": column.mode().iloc[0] if not column.mode().empty else None,
        "Min": column.min(),
        "Max": column.max(),
        "Range": column.max() - column.min(),
        "Standard Deviation": column.std(),
        "25th Percentile": column.quantile(0.25),
        "75th Percentile": column.quantile(0.75),
    }
    return stats

# ฟังก์ชันสำหรับวาดกราฟ Bar Chart
def plot_bar_chart(column):
    counts = column.value_counts()
    plt.figure(figsize=(8, 6))
    counts.plot(kind="bar", color=["blue", "green", "orange"])
    plt.title("Bar Chart")
    plt.xlabel(column.name)
    plt.ylabel("Frequency")
    st.pyplot(plt)

# ฟังก์ชันสำหรับวาดกราฟ Histogram
def plot_histogram(column):
    plt.figure(figsize=(8, 6))
    sns.histplot(column, kde=True, bins=20, color="blue")
    plt.title(f"Distribution of {column.name}")
    plt.xlabel(column.name)
    plt.ylabel("Frequency")
    st.pyplot(plt)

# อินเทอร์เฟซใน Streamlit
st.title("📊 Analysis Delay")

# เลือกการดำเนินการ
option = st.sidebar.selectbox(
    "🔧 **Choose an Action**",
    ["📋 View Statistics", "📊 Plot Bar Chart", "📉 Plot Histogram"]
)

# เลือกคอลัมน์จากข้อมูล
selected_column = st.sidebar.selectbox(
    "📂 **Select a Column**",
    data.columns
)

# แสดงผลตามตัวเลือก
if option == "📋 View Statistics":
    st.header("📋 Statistics")
    column = data[selected_column]
    if column.dtype in ["float64", "int64"]:  # ตรวจสอบว่าคอลัมน์เป็นตัวเลข
        stats = calculate_statistics(column)
        st.write(pd.DataFrame(stats, index=[selected_column]))
    else:
        st.error("⚠️ Please select a numeric column for statistics.")

elif option == "📊 Plot Bar Chart":
    st.header("📊 Bar Chart")
    column = data[selected_column]
    if column.dtype == "object":  # ตรวจสอบว่าคอลัมน์เป็นประเภทข้อความ
        plot_bar_chart(column)
    else:
        st.error("⚠️ Please select a categorical column for bar chart.")

elif option == "📉 Plot Histogram":
    st.header("📉 Histogram")
    column = data[selected_column]
    if column.dtype in ["float64", "int64"]:  # ตรวจสอบว่าคอลัมน์เป็นตัวเลข
        plot_histogram(column)
    else:
        st.error("⚠️ Please select a numeric column for histogram.")
