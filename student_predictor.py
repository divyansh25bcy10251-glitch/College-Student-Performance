import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Global variables
students_data = []
trained_models = {}
model_accuracies = {}

# Student class (Physics removed)
class Student:
    def __init__(self, name, roll, study_hours, attendance, prev_gpa, 
                 parent_edu, math, programming, extra):
        self.name = name
        self.roll = roll
        self.study_hours = float(study_hours)
        self.attendance = float(attendance)
        self.prev_gpa = float(prev_gpa)
        self.parent_edu = int(parent_edu)
        self.math = float(math)
        self.programming = float(programming)
        self.extra = int(extra)
        self.final_score = (self.math + self.programming) / 2
        self.grade = self.calculate_grade()
    
    def calculate_grade(self):
        score = self.final_score
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'
    
    def get_features(self):
        return [self.study_hours, self.attendance, self.prev_gpa * 25,
                self.parent_edu * 10, self.math, self.programming, 
                self.extra * 10]

# Generate sample dataset
def generate_sample_data():
    global students_data
    students_data = []
    
    names = ['Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 
             'Henry', 'Iris', 'Jack', 'Kate', 'Liam', 'Maya', 'Noah', 
             'Olivia', 'Peter', 'Quinn', 'Ruby', 'Sam', 'Tina', 'Uma', 
             'Victor', 'Wendy', 'Xavier', 'Yara', 'Zack', 'Anna', 'Ben', 
             'Clara', 'Dan']
    
    for i in range(30):
        study_hours = np.random.randint(5, 35)
        attendance = np.random.randint(70, 100)
        prev_gpa = round(np.random.uniform(2.0, 4.0), 2)
        parent_edu = np.random.randint(1, 5)
        extra = np.random.randint(0, 3)
        
        base = 50 + np.random.rand() * 30
        variation = np.random.rand() * 20 - 10
        
        math = min(100, max(40, int(base + variation + study_hours * 0.5)))
        programming = min(100, max(40, int(base + variation + prev_gpa * 8)))

        student = Student(names[i], f"2024{str(i+1).zfill(3)}", 
                         study_hours, attendance, prev_gpa, parent_edu,
                         math, programming, extra)
        students_data.append(student)
    
    return "✅ Generated 30 sample students successfully!"

# Add single student
def add_student(name, roll, study_hours, attendance, prev_gpa, 
                parent_edu, math, programming, extra):
    try:
        student = Student(name, roll, study_hours, attendance, prev_gpa,
                         parent_edu, math, programming, extra)
        students_data.append(student)
        return f"✅ Added {name} successfully! Total students: {len(students_data)}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# View dataset
def view_dataset():
    if not students_data:
        return pd.DataFrame()
    
    data = {
        'Name': [s.name for s in students_data],
        'Roll': [s.roll for s in students_data],
        'Study Hrs': [s.study_hours for s in students_data],
        'Attendance': [f"{s.attendance}%" for s in students_data],
        'Math': [s.math for s in students_data],
        'Programming': [s.programming for s in students_data],
        'Final Score': [round(s.final_score, 1) for s in students_data],
        'Grade': [s.grade for s in students_data]
    }
    
    return pd.DataFrame(data)

# Train models
def train_models(use_rf, use_dt, use_nb, use_knn):
    global trained_models, model_accuracies
    
    if len(students_data) < 10:
        return "❌ Need at least 10 students to train models!"
    
    if not any([use_rf, use_dt, use_nb, use_knn]):
        return "❌ Please select at least one algorithm!"
    
    X = np.array([s.get_features() for s in students_data])
    y = np.array([s.grade for s in students_data])
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    trained_models = {}
    model_accuracies = {}
    results = []
    
    if use_rf:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        accuracy = rf.score(X_test, y_test) * 100
        trained_models['Random Forest'] = (rf, le)
        model_accuracies['Random Forest'] = accuracy
        results.append(f"Random Forest: {accuracy:.2f}%")
    
    if use_dt:
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        accuracy = dt.score(X_test, y_test) * 100
        trained_models['Decision Tree'] = (dt, le)
        model_accuracies['Decision Tree'] = accuracy
        results.append(f"Decision Tree: {accuracy:.2f}%")
    
    if use_nb:
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        accuracy = nb.score(X_test, y_test) * 100
        trained_models['Naive Bayes'] = (nb, le)
        model_accuracies['Naive Bayes'] = accuracy
        results.append(f"Naive Bayes: {accuracy:.2f}%")
    
    if use_knn:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test) * 100
        trained_models['K-Nearest Neighbors'] = (knn, le)
        model_accuracies['K-Nearest Neighbors'] = accuracy
        results.append(f"KNN: {accuracy:.2f}%")
    
    return "✅ Training Complete!\n\nModel Accuracies:\n" + "\n".join(results)

# Predict performance
def predict_performance(study_hours, attendance, prev_gpa, parent_edu,
                       math, programming, extra):
    if not trained_models:
        return "❌ Please train models first!"
    
    features = np.array([[
        float(study_hours), float(attendance), float(prev_gpa) * 25,
        int(parent_edu) * 10, float(math), float(programming),
        int(extra) * 10
    ]])
    
    predictions = {}
    for model_name, (model, le) in trained_models.items():
        pred_encoded = model.predict(features)[0]
        pred_grade = le.inverse_transform([pred_encoded])[0]
        confidence = model_accuracies[model_name]
        predictions[model_name] = (pred_grade, confidence)
    
    grades = [p[0] for p in predictions.values()]
    final_pred = max(set(grades), key=grades.count)
    
    result = f"🎯 **Predicted Grade: {final_pred}**\n\n"
    result += "Model Predictions:\n"
    for model_name, (grade, conf) in predictions.items():
        result += f"• {model_name}: {grade} (Confidence: {conf:.1f}%)\n"
    
    avg_score = (float(math) + float(programming)) / 2
    result += f"\n📊 Expected Final Score: {avg_score:.1f}/100"
    
    return result

# Export dataset
def export_dataset():
    if not students_data:
        return None
    
    data = {
        'Name': [s.name for s in students_data],
        'Roll': [s.roll for s in students_data],
        'StudyHours': [s.study_hours for s in students_data],
        'Attendance': [s.attendance for s in students_data],
        'PrevGPA': [s.prev_gpa for s in students_data],
        'ParentEdu': [s.parent_edu for s in students_data],
        'Math': [s.math for s in students_data],
        'Programming': [s.programming for s in students_data],
        'Extra': [s.extra for s in students_data],
        'FinalScore': [s.final_score for s in students_data],
        'Grade': [s.grade for s in students_data]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('student_dataset.csv', index=False)
    return 'student_dataset.csv'

# Create Gradio Interface (THEME REMOVED)
with gr.Blocks(title="Student Performance ML Predictor") as demo:
    
    gr.Markdown("""
    # 🎓 Student Performance ML Predictor
    ### Educational Data Mining with Machine Learning | University AI Project
    """)
    
    with gr.Tabs():
        # Tab 1: Data Management
        with gr.Tab("📊 Data Management"):
            gr.Markdown("### Add Student Data")
            
            with gr.Row():
                name = gr.Textbox(label="Student Name", placeholder="John Doe")
                roll = gr.Textbox(label="Roll Number", placeholder="2024001")
            
            with gr.Row():
                study_hours = gr.Number(label="Study Hours/Week", value=15, minimum=0, maximum=168)
                attendance = gr.Number(label="Attendance %", value=85, minimum=0, maximum=100)
                prev_gpa = gr.Number(label="Previous GPA", value=3.0, minimum=0, maximum=4, step=0.01)
            
            with gr.Row():
                parent_edu = gr.Dropdown(choices=[("High School", 1), ("Bachelor's", 2), 
                                                  ("Master's", 3), ("PhD", 4)],
                                        label="Parent Education", value=2)
                extra = gr.Dropdown(choices=[("None", 0), ("1-2 Activities", 1), 
                                            ("3+ Activities", 2)],
                                   label="Extracurricular", value=1)
            
            with gr.Row():
                math = gr.Number(label="Math Score", value=75, minimum=0, maximum=100)
                programming = gr.Number(label="Programming Score", value=85, minimum=0, maximum=100)
            
            add_btn = gr.Button("➕ Add Student", variant="primary")
            add_output = gr.Textbox(label="Status", interactive=False)
            
            add_btn.click(
                fn=add_student,
                inputs=[name, roll, study_hours, attendance, prev_gpa, 
                       parent_edu, math, programming, extra],
                outputs=add_output
            )
            
            gr.Markdown("### Dataset Management")
            
            with gr.Row():
                gen_btn = gr.Button("🔄 Generate Sample Dataset (30 students)")
                view_btn = gr.Button("👁️ View Dataset")
                export_btn = gr.Button("💾 Export CSV")
            
            gen_output = gr.Textbox(label="Status", interactive=False)
            dataset_view = gr.Dataframe(label="Student Dataset")
            export_file = gr.File(label="Download Dataset")
            
            gen_btn.click(fn=generate_sample_data, outputs=gen_output)
            view_btn.click(fn=view_dataset, outputs=dataset_view)
            export_btn.click(fn=export_dataset, outputs=export_file)
        
        # Tab 2: Train Models
        with gr.Tab("🤖 Train Models"):
            gr.Markdown("""
            ### Machine Learning Model Training
            Select algorithms to train on the student dataset.
            """)
            
            with gr.Row():
                use_rf = gr.Checkbox(label="Random Forest", value=True)
                use_dt = gr.Checkbox(label="Decision Tree", value=True)
                use_nb = gr.Checkbox(label="Naive Bayes", value=True)
                use_knn = gr.Checkbox(label="K-Nearest Neighbors", value=True)
            
            train_btn = gr.Button("🚀 Train Selected Models", variant="primary")
            train_output = gr.Textbox(label="Training Results", lines=8, interactive=False)
            
            train_btn.click(
                fn=train_models,
                inputs=[use_rf, use_dt, use_nb, use_knn],
                outputs=train_output
            )
        
        # Tab 3: Predict Performance
        with gr.Tab("🔮 Predict Performance"):
            gr.Markdown("""
            ### Predict Student Grade
            Enter student data to predict final grade
            """)
            
            with gr.Row():
                pred_study = gr.Number(label="Study Hours/Week", value=15, minimum=0, maximum=168)
                pred_attendance = gr.Number(label="Attendance %", value=85, minimum=0, maximum=100)
                pred_gpa = gr.Number(label="Previous GPA", value=3.2, minimum=0, maximum=4, step=0.01)
            
            with gr.Row():
                pred_parent = gr.Dropdown(choices=[("High School", 1), ("Bachelor's", 2), 
                                                   ("Master's", 3), ("PhD", 4)],
                                         label="Parent Education", value=2)
                pred_extra = gr.Dropdown(choices=[("None", 0), ("1-2 Activities", 1), 
                                                 ("3+ Activities", 2)],
                                        label="Extracurricular", value=1)
            
            with gr.Row():
                pred_math = gr.Number(label="Math Score", value=78, minimum=0, maximum=100)
                pred_prog = gr.Number(label="Programming Score", value=85, minimum=0, maximum=100)
            
            predict_btn = gr.Button("🔮 Predict Grade", variant="primary")
            predict_output = gr.Textbox(label="Prediction Results", lines=10, interactive=False)
            
            predict_btn.click(
                fn=predict_performance,
                inputs=[pred_study, pred_attendance, pred_gpa, pred_parent,
                       pred_math, pred_prog, pred_extra],
                outputs=predict_output
            )
        
        # Tab 4: About
        with gr.Tab("💡 About"):
            gr.Markdown("""
            ## 📚 Machine Learning Algorithms
            
            - **Random Forest:** Ensemble method
            - **Decision Tree:** Tree-based classification
            - **Naive Bayes:** Probabilistic classifier
            - **K-Nearest Neighbors:** Instance-based learning
            
            ## 📊 Features (7 variables)
            
            1. Study hours per week
            2. Attendance percentage
            3. Previous GPA
            4. Parent education
            5. Math score
            6. Programming score
            7. Extracurricular activities
            """)

# Launch
if __name__ == "__main__":
    demo.launch(share=True)
