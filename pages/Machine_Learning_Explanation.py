import streamlit as st

st.title("🔭 การพยากรณ์อุณหภูมิของดาวฤกษ์")

st.write("โมเดล Machine Learning นี้ใช้ **SVM (Support Vector Machine)** และ **Random Forest** \nเพื่อทำนายอุณหภูมิของดาวฤกษ์ โดยอ้างอิงจากคุณสมบัติต่างๆ เช่น \n**ความสว่าง (Luminosity), รัศมี (Radius), ค่าความส่องสว่างสัมบูรณ์ (Absolute Magnitude), สี (Color), \nและประเภทสเปกตรัม (Spectral Class)**")

st.header("🧠 วิธีการทำงานของโมเดล")

st.subheader("1️⃣ Support Vector Machine (SVM)")
st.write("SVM เป็นอัลกอริธึมที่ใช้ในการจำแนกหรือคาดการณ์ค่าต่างๆ โดยอาศัยแนวคิดของ **Hyperplane** \nซึ่งใช้แบ่งกลุ่มข้อมูลออกจากกัน โมเดล SVM ของเราจะใช้พารามิเตอร์ของดาวในการทำนาย \nอุณหภูมิ โดยหาเส้นแบ่งที่เหมาะสมที่สุดเพื่อลดข้อผิดพลาดในการทำนาย")

st.subheader("2️⃣ Random Forest")
st.write("Random Forest เป็นอัลกอริธึมที่ใช้ **การรวมกันของหลายๆ ต้นไม้ตัดสินใจ (Decision Trees)** \nในการพยากรณ์ค่า เพื่อให้ได้ผลลัพธ์ที่แม่นยำและลด Overfitting โมเดลจะเรียนรู้จากข้อมูลจำนวนมาก \nและสร้างต้นไม้หลายต้นเพื่อพยากรณ์อุณหภูมิของดาว")

st.header("🔹 วิธีใช้งานโมเดล")
st.write("คุณสามารถใช้โมเดลของเราได้ผ่านหน้าเดโม ซึ่งให้คุณ **ป้อนค่าพารามิเตอร์ของดาว** เช่น ความสว่าง, รัศมี, \nสี, และประเภทสเปกตรัม จากนั้นระบบจะคำนวณและแสดงผลอุณหภูมิที่คาดการณ์จากโมเดล SVM และ Random Forest\n")

