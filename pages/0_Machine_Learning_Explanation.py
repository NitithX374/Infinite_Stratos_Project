import streamlit as st
import pandas as pd

# Title of the Streamlit app
st.title("🔭 การพยากรณ์อุณหภูมิของดาวฤกษ์")

# Explanation of the model
st.write("""
โมเดล Machine Learning นี้ใช้ **SVM (Support Vector Machine)** และ **Random Forest** 
เพื่อทำนายอุณหภูมิของดาวฤกษ์ โดยอ้างอิงจากคุณสมบัติต่างๆ เช่น 
**ความสว่าง (Luminosity), รัศมี (Radius), ค่าความส่องสว่างสัมบูรณ์ (Absolute Magnitude), สี (Color), 
และประเภทสเปกตรัม (Spectral Class)**
""")

# Model description
st.header("🧠 วิธีการทำงานของโมเดล")

# SVM explanation
st.subheader("1️⃣ Support Vector Machine (SVM)")
st.write("""
SVM เป็นอัลกอริธึมที่ใช้ในการจำแนกหรือคาดการณ์ค่าต่างๆ โดยอาศัยแนวคิดของ **Hyperplane** 
ซึ่งใช้แบ่งกลุ่มข้อมูลออกจากกัน โมเดล SVM ของเราจะใช้พารามิเตอร์ของดาวในการทำนาย 
อุณหภูมิ โดยหาเส้นแบ่งที่เหมาะสมที่สุดเพื่อลดข้อผิดพลาดในการทำนาย
""")

# Random Forest explanation
st.subheader("2️⃣ Random Forest")
st.write("""
Random Forest เป็นอัลกอริธึมที่ใช้ **การรวมกันของหลายๆ ต้นไม้ตัดสินใจ (Decision Trees)** 
ในการพยากรณ์ค่า เพื่อให้ได้ผลลัพธ์ที่แม่นยำและลด Overfitting โมเดลจะเรียนรู้จากข้อมูลจำนวนมาก 
และสร้างต้นไม้หลายต้นเพื่อพยากรณ์อุณหภูมิของดาว
""")

# Dataset description
st.header("📊 ตัวอย่างข้อมูลของดาวฤกษ์")

# Creating the dataset table from provided data
data = {
    "Temperature": [3068, 3042, 2600, 2800, 1939, 2840, 2637, 2600, 2650, 2700, 3600],
    "L": [0.0024, 0.0005, 0.0003, 0.0002, 0.000138, 0.00065, 0.00073, 0.0004, 0.00069, 0.00018, 0.0029],
    "R": [0.17, 0.1542, 0.102, 0.16, 0.103, 0.11, 0.127, 0.096, 0.11, 0.13, 0.51],
    "A_M": [16.12, 16.6, 18.7, 16.65, 20.06, 16.98, 17.22, 17.4, 17.45, 16.05, 10.69],
    "Color": ["Red", "Red", "Red", "Red", "Red", "Red", "Red", "Red", "Red", "Red", "Red"],
    "Spectral_Class": ["M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M"],
    "Type": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the dataset table
st.write("ข้อมูลตัวอย่างของดาวฤกษ์ที่ใช้ในการฝึกโมเดล")
st.dataframe(df)

# Additional explanation
st.write("""
ในตารางนี้จะเห็นว่ามีการใช้พารามิเตอร์ต่างๆ ของดาวฤกษ์ เช่น **Temperature (อุณหภูมิ), 
L (ความสว่าง), R (รัศมี), A_M (ค่าความส่องสว่างสัมบูรณ์), Color (สี), Spectral Class (ประเภทสเปกตรัม)** 
และ **Type (ประเภท)** เพื่อใช้ในการทำนายอุณหภูมิของดาว
""")
