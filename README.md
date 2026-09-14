<p align="center"> <img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,50:764ba2,100:f093fb&height=220&section=header&text=Student%20Career%20Success%20Prediction&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=35" width="100%"/> </p>

<p align="center">

<p align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&color=4CAF50&center=true&vCenter=true&width=700&lines=Student+Career+Success+Prediction;Machine+Learning+%7C+Data+Analytics;Predicting+Student+Placement+Success;Built+with+Python+%26+Streamlit" alt="Typing SVG" />

</p>

<p align="center">
  <b>🚀 An End-to-End Machine Learning Project for Predicting Student Placement Success</b>
</p>

<p align="center">






</p>

---

## 🌟 Project Overview

> **Can we predict whether a student is likely to get placed based on their academic performance, technical skills, projects, internships, communication skills, and other career-related factors?**

This project uses **Machine Learning** to predict a student's placement status based on multiple academic, technical, professional, and personal development features.

The project takes a student's profile as input and estimates the probability of successful placement.

### 🎯 Prediction Target

**Placement Status**

* 🟢 **Placed**
* 🔴 **Not Placed**

The application also provides a **placement probability score** to give a more detailed understanding of the prediction.

---

# ✨ What Makes This Project Different?

This is not just a basic ML classification project.

It combines:

```text
📊 Data Analysis
      ↓
🧹 Data Preprocessing
      ↓
🔄 Feature Transformation
      ↓
⚖️ Class Imbalance Handling
      ↓
🤖 Multiple ML Models
      ↓
📈 Model Evaluation
      ↓
🎯 Probability Prediction
      ↓
🌐 Streamlit Web Application
```

The final project provides an interactive interface where users can enter student information and instantly receive a prediction.

---

# 🚀 Live Application

> 🌐 **Live Demo: https://student-career-success-prediction-n4xvvdute4gcwqjrwgyjdp.streamlit.app/

The application is built using **Streamlit** and is designed as an interactive ML prediction dashboard.

---

# 🧠 Machine Learning Models

Four classification algorithms were evaluated:

| Model                  | Purpose                        |
| ---------------------- | ------------------------------ |
| 🔵 Logistic Regression | Baseline linear classification |
| 🌳 Decision Tree       | Rule-based classification      |
| 🌲 Random Forest       | Ensemble tree-based model      |
| 🚀 Gradient Boosting   | Sequential ensemble learning   |

The final model is selected based on **F1 Score**, which is particularly useful when the target classes are imbalanced.

---

# ⚖️ Handling Class Imbalance

The dataset contains an imbalance between placed and non-placed students.

To address this problem, the project uses:

### SMOTE — Synthetic Minority Over-sampling Technique

SMOTE generates synthetic samples for the minority class instead of simply duplicating existing observations.

This helps the model learn both classes more effectively.

```text
Original Data
      ↓
Train / Test Split
      ↓
Preprocessing
      ↓
SMOTE
      ↓
Machine Learning Model
      ↓
Evaluation
```

---

# ⚙️ Data Preprocessing Pipeline

The project uses a structured preprocessing pipeline.

### 🔢 Numerical Features

Numerical columns are processed using:

* Missing-value imputation
* Median strategy
* StandardScaler

### 🔤 Categorical Features

Categorical columns are processed using:

* Missing-value imputation
* Most-frequent strategy
* One-Hot Encoding

### 🧩 Pipeline Architecture

```text
Raw Student Data
       │
       ▼
┌─────────────────────┐
│ Data Cleaning       │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Numerical Features  │
│ Median Imputation   │
│ Standard Scaling    │
└──────────┬──────────┘
           │
           ├──────────────┐
           │              │
           ▼              ▼
 Numerical            Categorical
 Features             Features
                           │
                           ▼
                    One-Hot Encoding
                           │
                           ▼
                      SMOTE
                           │
                           ▼
                    ML Classifier
                           │
                           ▼
                  Placement Prediction
```

---

# 📋 Input Features

The model uses a wide range of student-related features.

### 🎓 Academic Features

* Age
* University Year
* Major
* Attendance Percentage
* Study Hours Per Week
* CGPA
* Academic Performance

### 💻 Technical Features

* Programming Skill
* Projects Completed
* Certifications
* Hackathons
* GitHub Profile

### 💼 Professional Features

* Internships
* Leadership Experience
* LinkedIn Profile
* Resume Score
* Employability Score

### 🗣️ Personal & Communication Features

* Communication Skills
* Teamwork
* Problem Solving
* English Proficiency
* Interview Score

---

# 📊 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

### 🏆 Model Selection

The final model is selected according to its **F1 Score**.

> F1 Score provides a balance between Precision and Recall and is useful when the dataset has class imbalance.

---

# 🎯 Prediction System

The Streamlit application allows users to enter student information through an interactive interface.

After clicking the prediction button, the application displays:

### 🟢 Placement Prediction

Example:

```text
PLACEMENT STATUS

🟢 LIKELY TO BE PLACED

Placement Probability: 92.4%
```

The application also provides a probability visualization and student profile analysis.

---

# 📈 Interactive Dashboard

The application contains multiple sections:

### 🏠 Home

Provides an overview of the project and prediction system.

### 🎯 Placement Prediction

Users can enter student information and receive a prediction.

### 📊 Model Performance

Displays:

* Model comparison
* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

### 🧠 Feature Importance

Shows which features have the strongest influence on model predictions.

### 🔄 ML Workflow

Explains the complete machine learning pipeline.

### 👨‍💻 About Project

Provides project and developer information.

---

# 🖥️ Application Preview

> 📸 Screenshots of the Streamlit application will be added here.

You can add your screenshots later like this:

# Home Page
<img width="1897" height="897" alt="Screenshot 2026-09-13 122026" src="https://github.com/user-attachments/assets/e7551cae-3504-49f5-b48c-8ff186ae7940" />


# Prediction Page
<img width="1887" height="856" alt="Screenshot 2026-09-13 122108" src="https://github.com/user-attachments/assets/79188201-82bd-4422-a876-0bf6e677517b" />


# Model Performance
<img width="1885" height="877" alt="Screenshot 2026-09-13 122138" src="https://github.com/user-attachments/assets/066ba31d-240b-4f83-b2af-df7aa462262c" />

```

---

# 🛠️ Tech Stack

### Programming

### Data Analysis

\

### Machine Learning

\

### Visualization

### Deployment

### Model Persistence

---

# 📂 Project Structure

```text
student-career-success-prediction/
│
├── 📄 app_new.py
│
├── 📄 train_model.py
│
├── 📄 README.md
│
├── 📄 requirements.txt
│
├── 🤖 student_placement_model.pkl
├── 📦 feature_names.pkl
├── 📦 input_options.pkl
├── 📦 model_info.pkl
├── 📦 model_comparison.pkl
├── 📦 feature_importance.pkl
├── 📦 confusion_matrix.pkl
├── 📦 classification_report.pkl
├── 📦 training_metadata.pkl
│
└── 📁 images/
    ├── home.png
    ├── prediction.png
    └── performance.png
```

---

# 🚀 How to Run Locally

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/anandpal3244-coder/student-career-success-prediction.git
```

## 2️⃣ Open the Project

```bash
cd student-career-success-prediction
```

## 3️⃣ Create a Virtual Environment

```bash
python -m venv .venv
```

## 4️⃣ Activate the Environment

### Windows

```bash
.venv\Scripts\activate
```

### Mac / Linux

```bash
source .venv/bin/activate
```

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 6️⃣ Run the Streamlit Application

```bash
streamlit run app_new.py
```

The application will open in your browser.

---

# 📦 Requirements

The project requires the following major libraries:

```text
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
joblib
plotly
```

---

# 🔬 Machine Learning Workflow

```text
                 ┌──────────────────┐
                 │   Student Data   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Data Cleaning    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Train/Test Split │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Preprocessing    │
                 │                  │
                 │ Imputation       │
                 │ Scaling          │
                 │ Encoding         │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │      SMOTE       │
                 └────────┬─────────┘
                          │
                          ▼
             ┌───────────────────────────┐
             │     ML Model Training     │
             │                           │
             │ Logistic Regression       │
             │ Decision Tree             │
             │ Random Forest             │
             │ Gradient Boosting         │
             └─────────────┬─────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Model Evaluation│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Best Model       │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Streamlit App    │
                 └────────┬─────────┘
                          │
                          ▼
                 🎯 Placement Prediction
```

---

# 💡 Key Learning Outcomes

Through this project, I worked on:

* ✅ Exploratory Data Analysis
* ✅ Data Cleaning
* ✅ Missing Value Handling
* ✅ Numerical Feature Scaling
* ✅ Categorical Feature Encoding
* ✅ Train-Test Split
* ✅ SMOTE for Class Imbalance
* ✅ Multiple Classification Algorithms
* ✅ Model Evaluation
* ✅ Confusion Matrix
* ✅ Classification Report
* ✅ Feature Importance
* ✅ Probability Prediction
* ✅ Model Serialization using Joblib
* ✅ Streamlit Application Development
* ✅ Interactive Data Visualization
* ✅ End-to-End ML Project Development

---

# 🎯 Business Use Case

This project can be useful for educational institutions, placement cells, and career-support platforms.

Potential applications include:

```text
Student Data
     ↓
Career Analysis
     ↓
Placement Probability
     ↓
Identify Students Needing Support
     ↓
Targeted Skill Development
     ↓
Improved Placement Readiness
```

For example, students with lower predicted placement probability could receive additional support in:

* 💻 Technical skills
* 🗣️ Communication
* 📄 Resume building
* 🎤 Interview preparation
* 💼 Internship opportunities
* 🧠 Problem-solving skills

---

# ⚠️ Important Note

This project is intended for **educational and analytical purposes**.

A machine learning prediction should not be treated as a guaranteed outcome for an individual student's career.

Real-world placement decisions involve many factors that may not be captured by the dataset.

---

# 👨‍💻 Developer

## Anand Kumar

🎯 **Entry-Level Data Analyst**

🎓 **BA Geography Honours — Magadh University, 2023**

📊 **Data Analytics Training & Certification — iScale**

### Technical Skills

```text
Python • SQL • Excel • Pandas • NumPy
Power BI • Tableau • Matplotlib • Seaborn
Machine Learning • EDA • Statistics
Data Cleaning • Data Visualization
```

---

# 🔗 Connect With Me

<p align="center">

<a href="https://github.com/anandpal3244-coder">
<img src="https://img.shields.io/badge/GitHub-Anand%20Kumar-181717?style=for-the-badge&logo=github" />
</a>

<a href="https://www.linkedin.com/in/anand-pal-6a657b393/">
<img src="https://img.shields.io/badge/LinkedIn-Anand%20Kumar-0A66C2?style=for-the-badge&logo=linkedin" />
</a>

</p>

---

# ⭐ If You Like This Project

If you find this project useful or interesting:

⭐ **Star this repository**

🍴 **Fork the repository**

💬 **Share your feedback**

🤝 **Connect with me on LinkedIn**

---

<p align="center">

### 🚀 Built with Python • Machine Learning • Streamlit

**Made with ❤️ by Anand Kumar**

</p>

<p align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer"/>

</p>
