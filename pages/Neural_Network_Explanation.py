import streamlit as st
import pandas as pd

# Title of the Streamlit app
st.title("🛰️ การจำแนกภาพถ่ายดาวเทียมด้วย CNN")

# Explanation of the model
st.write("""
โมเดลของเราสามารถจำแนกประเภทของพื้นที่จากภาพถ่ายดาวเทียม เช่น 
**เมฆ (Cloudy), ทะเลทราย (Desert), พื้นที่สีเขียว (Green Area), และแหล่งน้ำ (Water)** 
โดยใช้ **Convolutional Neural Networks (CNNs)**
""")

# Model description
st.header("🧠 วิธีการทำงานของ CNN")

# Convolution Layer explanation
st.subheader("1️⃣ Convolution Layer")
st.write("""
เลเยอร์นี้ใช้ **ตัวกรอง (Filters)** เพื่อตรวจจับลักษณะเฉพาะในภาพ เช่น ขอบ, พื้นผิว, 
และรูปแบบต่างๆ ของภาพถ่ายดาวเทียม
""")

# Pooling Layer explanation
st.subheader("2️⃣ Pooling Layer")
st.write("""
เลเยอร์นี้ช่วยลดขนาดของข้อมูล แต่ยังคงรักษาลักษณะสำคัญของภาพไว้ ซึ่งช่วยให้โมเดลทำงานได้เร็วขึ้น 
และลดโอกาส Overfitting
""")

# Fully Connected Layer explanation
st.subheader("3️⃣ Fully Connected Layer")
st.write("""
ข้อมูลที่ผ่านการประมวลผลจะถูกแปลงเป็นเวกเตอร์และป้อนเข้าสู่ **Dense Layer** เพื่อทำนายประเภทของพื้นที่จากภาพถ่าย
""")

# Dataset description
st.header("📊 ตัวอย่างข้อมูลภาพดาวเทียม")

# Creating a sample dataset of satellite images
data = {
    "Image ID": [1, 2, 3, 4, 5],
    "Image": ["Cloudy Image", "Desert Image", "Green Area Image", "Water Image", "Mixed Image"],
    "Category": ["Cloudy", "Desert", "Green Area", "Water", "Mixed"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the dataset table
st.write("ข้อมูลตัวอย่างของภาพถ่ายดาวเทียมที่ใช้ในการฝึกโมเดล")
st.dataframe(df)

# Additional explanation
st.write("""
ในตารางนี้จะเห็นว่าโมเดลของเรามีการจำแนกประเภทของพื้นที่จากภาพดาวเทียมตามลักษณะต่างๆ เช่น 
**เมฆ (Cloudy), ทะเลทราย (Desert), พื้นที่สีเขียว (Green Area), และแหล่งน้ำ (Water)** 
โดยจะใช้ CNN ในการจำแนกประเภท
""")
