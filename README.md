# 🎓 Student Performance ML Predictor

A simple Gradio web app that uses classic machine learning algorithms to predict student grades (A–F) from study habits, attendance, GPA, parent education, extracurriculars, and subject scores. Perfect as a university AI/ML mini project or demo. [attached_file:21]

---

## 🚀 Features

- Add students manually or auto-generate 30 sample records. [attached_file:21]  
- View data in a table and export it as `student_dataset.csv`. [attached_file:21]  
- Train Random Forest, Decision Tree, Naive Bayes, and KNN on the dataset. [attached_file:21]  
- See accuracy of each model and compare performance. [attached_file:21]  
- Predict grades for new students with an ensemble-style summary and expected final score. [attached_file:21]  
- Clean, tabbed Gradio UI: Data, Train, Predict, About. [attached_file:21]  

---

## 🛠 Tech Stack

- **Language:** Python 3.7+  
- **ML & Data:** scikit-learn, pandas, numpy  
- **UI:** Gradio web interface  

---

## 📦 Installation

pip install gradio scikit-learn pandas numpy


Save the main script as `student_predictor.py` in a folder (e.g., `student-ml-predictor/`). [attached_file:21]

---

## ▶️ Run the App

python student_predictor.py


Open the shown local URL (e.g., `http://127.0.0.1:7860`) or share the temporary public Gradio link. [attached_file:21]

---

## ✅ Basic Usage Flow

1. Go to **📊 Data Management**  
   - Generate sample dataset or add students manually. [attached_file:21]  
2. Go to **🤖 Train Models**  
   - Select algorithms and click **Train Selected Models**. [attached_file:21]  
3. Go to **🔮 Predict Performance**  
   - Enter student details and click **Predict Grade** to see results. [attached_file:21]  

---

## 📁 Project Structure

student-ml-predictor/
├── student_predictor.py # Main app
└── README.md # This file

