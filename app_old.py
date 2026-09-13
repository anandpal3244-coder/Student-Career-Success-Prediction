import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Student Career Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.10), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(14,165,233,0.10), transparent 30%),
        #0b1020;
    color: #f8fafc;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 45px;
    border-radius: 28px;
    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.22),
            rgba(14,165,233,0.12)
        );
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 20px 60px rgba(0,0,0,0.30);
    animation: fadeIn 0.8s ease;
}

.hero h1 {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
    color: #cbd5e1;
    line-height: 1.7;
}

.card {
    padding: 24px;
    border-radius: 20px;
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 35px rgba(0,0,0,0.20);
    transition: all 0.3s ease;
    animation: fadeIn 0.7s ease;
}

.card:hover {
    transform: translateY(-5px);
    border-color: rgba(99,102,241,0.45);
    box-shadow: 0 18px 45px rgba(0,0,0,0.30);
}

.metric-card {
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(
        145deg,
        rgba(255,255,255,0.07),
        rgba(255,255,255,0.025)
    );
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
    margin-top: 5px;
}

.section-title {
    font-size: 30px;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 15px;
}

.success-box {
    padding: 30px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        rgba(34,197,94,0.18),
        rgba(16,185,129,0.08)
    );
    border: 1px solid rgba(34,197,94,0.35);
    text-align: center;
    animation: fadeIn 0.8s ease;
}

.warning-box {
    padding: 30px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        rgba(245,158,11,0.18),
        rgba(234,88,12,0.08)
    );
    border: 1px solid rgba(245,158,11,0.35);
    text-align: center;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

div.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    padding: 0.65rem 1.2rem;
    transition: all 0.25s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD FILES
# =========================================================

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(
        BASE_DIR / "student_placement_model.pkl"
    )



@st.cache_resource
def load_features():
    return joblib.load(
        BASE_DIR / "feature_names.pkl"
    )


@st.cache_data
def load_model_info():
    return joblib.load(
        BASE_DIR / "model_info.pkl"
    )


@st.cache_data
def load_model_comparison():
    return pd.read_pickle(
        BASE_DIR / "model_comparison.pkl"
    )


@st.cache_data
def load_feature_importance():
    return pd.read_pickle(
        BASE_DIR / "feature_importance.pkl"
    )


@st.cache_data
def load_confusion_matrix():
    return joblib.load(
        BASE_DIR / "confusion_matrix.pkl"
    )


@st.cache_data
def load_classification_report():
    return pd.read_pickle(
        BASE_DIR / "classification_report.pkl"
    )


try:

    model = load_model()

    feature_names = load_features()

    model_info = load_model_info()

    model_comparison = load_model_comparison()

    feature_importance = load_feature_importance()

    confusion_matrix = load_confusion_matrix()

    classification_report_df = load_classification_report()

except Exception as e:

    st.error(
        "Required model files are missing or could not be loaded."
    )

    st.exception(e)

    st.stop()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <div style="text-align:center;padding:10px;">
        <div style="font-size:48px;">🎓</div>
        <h2>Anand Kumar</h2>
        <p style="color:#94a3b8;">
            Student Career Success Predictor
        </p>
        <p style="color:#818cf8;font-size:13px;">
            Data Analyst
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Placement Prediction",
        "📊 Model Performance",
        "⭐ Feature Importance",
        "🔄 ML Workflow",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    **Final Model**

    `{model_info['model_name']}`

    **F1 Score:**  
    `{model_info['f1_score']:.3f}`
    """
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">

    <h1>🎓 Student Career Success Predictor</h1>

    <p>
    An intelligent machine learning application that estimates
    a student's placement success probability using academic,
    technical, professional and communication-related attributes.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Project at a Glance</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("🤖", "Final Model", model_info["model_name"]),
        ("🎯", "Accuracy", f"{model_info['accuracy']:.2%}"),
        ("📈", "F1 Score", f"{model_info['f1_score']:.2%}"),
        ("🏆", "ROC-AUC", f"{model_info['roc_auc']:.2%}")
    ]

    for col, (icon, label, value) in zip(
        [c1, c2, c3, c4],
        metrics
    ):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:28px;">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">How It Works</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(5)

    steps = [
        ("01", "Student Data", "Academic & skill profile"),
        ("02", "Preprocessing", "Encoding & preparation"),
        ("03", "SMOTE", "Class balancing"),
        ("04", "ML Model", "Prediction engine"),
        ("05", "Prediction", "Placement probability")
    ]

    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <div style="font-size:14px;color:#818cf8;">
                        STEP {num}
                    </div>
                    <h3>{title}</h3>
                    <p style="color:#94a3b8;">
                        {desc}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Placement Prediction":

    st.markdown(
        '<div class="section-title">🔮 Placement Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the student's profile to estimate placement success."
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=40,
            value=22
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        university_year = st.selectbox(
            "University Year",
            ["1st Year", "2nd Year", "3rd Year", "4th Year"]
        )

        major = st.text_input(
            "Major",
            value="Computer Science"
        )

        attendance = st.number_input(
            "Attendance Percentage",
            min_value=0.0,
            max_value=100.0,
            value=80.0
        )

        study_hours = st.number_input(
            "Study Hours Per Week",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )

    with col2:
        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.5
        )

        academic_performance = st.selectbox(
            "Academic Performance",
            ["Low", "Medium", "High"]
        )

        programming_skill = st.number_input(
            "Programming Skill",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        projects = st.number_input(
            "Projects Completed",
            min_value=0,
            max_value=30,
            value=3
        )

        certifications = st.number_input(
            "Certifications",
            min_value=0,
            max_value=30,
            value=2
        )

        hackathons = st.number_input(
            "Hackathons",
            min_value=0,
            max_value=20,
            value=1
        )

        github = st.selectbox(
            "GitHub Profile",
            ["Yes", "No"]
        )

    with col3:
        internships = st.number_input(
            "Internships",
            min_value=0,
            max_value=10,
            value=1
        )

        leadership = st.selectbox(
            "Leadership Experience",
            ["Yes", "No"]
        )

        linkedin = st.selectbox(
            "LinkedIn Profile",
            ["Yes", "No"]
        )

        resume_score = st.number_input(
            "Resume Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        communication = st.number_input(
            "Communication Skills",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        teamwork = st.number_input(
            "Teamwork",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        problem_solving = st.number_input(
            "Problem Solving",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        english = st.selectbox(
            "English Proficiency",
            ["Basic", "Intermediate", "Advanced"]
        )

    interview_score = st.number_input(
        "Interview Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    employability_score = st.number_input(
        "Employability Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    predict_button = st.button(
        "🚀 Predict Placement Success",
        use_container_width=True
    )

    if predict_button:

        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "University_Year": university_year,
            "Major": major,
            "Attendance_Percentage": attendance,
            "Study_Hours_Per_Week": study_hours,
            "CGPA": cgpa,
            "Academic_Performance": academic_performance,
            "Programming_Skill": programming_skill,
            "Projects_Completed": projects,
            "Certifications": certifications,
            "Hackathons": hackathons,
            "GitHub_Profile": github,
            "Internships": internships,
            "Leadership_Experience": leadership,
            "LinkedIn_Profile": linkedin,
            "Resume_Score": resume_score,
            "Communication_Skills": communication,
            "Teamwork": teamwork,
            "Problem_Solving": problem_solving,
            "English_Proficiency": english,
            "Interview_Score": interview_score,
            "Employability_Score": employability_score
        }])

        categorical_cols = [
            "Gender",
            "University_Year",
            "Major",
            "Academic_Performance",
            "GitHub_Profile",
            "Leadership_Experience",
            "LinkedIn_Profile",
            "English_Proficiency"
        ]

        input_encoded = pd.get_dummies(
            input_data,
            columns=categorical_cols,
            drop_first=True,
            dtype=int
        )

        input_encoded = input_encoded.reindex(
            columns=feature_names,
            fill_value=0
        )

        # IMPORTANT:
        # The currently saved model was trained without the scaler,
        # so prediction must use the same unscaled feature format.

        prediction = model.predict(
            input_encoded
        )[0]

        probability = model.predict_proba(
            input_encoded
        )[0][1]

        percentage = float(probability * 100)

        st.write("DEBUG - Raw Probability:", probability)
        st.write("DEBUG - Placement Probability:", f"{percentage:.6f}%")

        st.markdown("---")

        if percentage >= 75:

            st.markdown(
                f"""
                <div class="success-box">

                <div style="font-size:55px;">🎉</div>

                <h1>High Placement Potential</h1>

                <h2>{percentage:.1f}%</h2>

                <p>
                The model estimates a high probability of successful placement.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.balloons()

        elif percentage >= 50:

            st.markdown(
                f"""
                <div class="card">

                <div style="font-size:55px;">📊</div>

                <h1>Moderate Placement Potential</h1>

                <h2>{percentage:.1f}%</h2>

                <p>
                The student shows moderate placement potential.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="warning-box">

                <div style="font-size:55px;">📚</div>

                <h1>Needs Improvement</h1>

                <h2>{percentage:.1f}%</h2>

                <p>
                The model estimates a lower probability of successful placement.
                Focus on improving academic, technical and professional skills.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        # Probability gauge

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=percentage,
                title={
                    "text": "Placement Success Probability"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "thickness": 0.25
                    },
                    "steps": [
                        {
                            "range": [0, 50]
                        },
                        {
                            "range": [50, 75]
                        },
                        {
                            "range": [75, 100]
                        }
                    ]
                }
            )
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Student skill radar

        categories = [
            "Programming",
            "Communication",
            "Teamwork",
            "Problem Solving",
            "Resume",
            "Interview"
        ]

        values = [
            programming_skill,
            communication,
            teamwork,
            problem_solving,
            resume_score,
            interview_score
        ]

        fig_radar = go.Figure()

        fig_radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Student Profile'
            )
        )

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.markdown(
            '<div class="section-title">Student Skill Profile</div>',
            unsafe_allow_html=True
        )

        st.plotly_chart(
            fig_radar,
            use_container_width=True
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="section-title">📊 Model Performance</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    metric_values = [
        ("Accuracy", model_info["accuracy"]),
        ("Precision", model_info["precision"]),
        ("Recall", model_info["recall"]),
        ("F1 Score", model_info["f1_score"]),
        ("ROC-AUC", model_info["roc_auc"])
    ]

    for col, (label, value) in zip(
        [c1, c2, c3, c4, c5],
        metric_values
    ):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {value:.2%}
                    </div>

                    <div class="metric-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True
    )

    comparison_long = model_comparison.melt(
        id_vars="Model",
        value_vars=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        comparison_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".2f"
    )

    fig.update_layout(
        height=500,
        yaxis_range=[0, 1],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    cm_df = pd.DataFrame(
        confusion_matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto"
    )

    fig_cm.update_layout(
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Classification Report</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        classification_report_df.round(3),
        use_container_width=True
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

elif page == "⭐ Feature Importance":

    st.markdown(
        '<div class="section-title">⭐ Feature Importance</div>',
        unsafe_allow_html=True
    )

    importance_column = (
        "Absolute_Importance"
        if "Absolute_Importance" in feature_importance.columns
        else "Importance"
    )

    top_features = feature_importance.head(15).copy()

    top_features = top_features.sort_values(
        by=importance_column,
        ascending=True
    )

    fig = px.bar(
        top_features,
        x=importance_column,
        y="Feature",
        orientation="h",
        text_auto=".3f"
    )

    fig.update_layout(
        height=650,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Top Predictive Features</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        feature_importance.head(15),
        use_container_width=True
    )


# =========================================================
# WORKFLOW
# =========================================================

elif page == "🔄 ML Workflow":

    st.markdown(
        '<div class="section-title">🔄 Machine Learning Workflow</div>',
        unsafe_allow_html=True
    )

    workflow = [
        ("01", "Data Collection", "Student career success dataset"),
        ("02", "Data Cleaning", "Missing values and data preparation"),
        ("03", "Encoding", "Categorical variables converted to numerical format"),
        ("04", "Train-Test Split", "80% training and 20% testing"),
        ("05", "SMOTE", "Balance the training classes"),
        ("06", "Model Training", "Multiple classification algorithms"),
        ("07", "Hyperparameter Tuning", "Optimize model parameters"),
        ("08", "Evaluation", "Accuracy, Precision, Recall, F1 and ROC-AUC"),
        ("09", "Final Model", "Select best performing model"),
        ("10", "Deployment", "Interactive Streamlit application")
    ]

    for num, title, description in workflow:

        st.markdown(
            f"""
            <div class="card" style="margin-bottom:15px;">

                <div style="
                    display:flex;
                    gap:20px;
                    align-items:center;
                ">

                    <div style="
                        font-size:20px;
                        font-weight:800;
                        color:#818cf8;
                        min-width:45px;
                    ">
                        {num}
                    </div>

                    <div>

                        <h3 style="margin:0;">
                            {title}
                        </h3>

                        <p style="
                            color:#94a3b8;
                            margin-top:5px;
                        ">
                            {description}
                        </p>

                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="section-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h2>🎓 Student Career Success Prediction</h2>

    <p>
    This project uses machine learning classification techniques
    to predict whether a student is likely to achieve successful
    placement based on academic performance, technical skills,
    projects, internships, communication and professional profile.
    </p>

    <h3>🎯 Target Variable</h3>
    <p><b>Placement_Status</b></p>

    <p>
    The target represents whether a student was successfully placed
    or not.
    </p>

    <h3>🤖 Machine Learning</h3>

    <ul>
        <li>Logistic Regression</li>
        <li>Decision Tree</li>
        <li>Random Forest</li>
        <li>Gradient Boosting</li>
        <li>SMOTE for class balancing</li>
        <li>Hyperparameter tuning</li>
    </ul>

    <h3>📊 Evaluation</h3>

    <ul>
        <li>Accuracy</li>
        <li>Precision</li>
        <li>Recall</li>
        <li>F1 Score</li>
        <li>ROC-AUC</li>
        <li>Confusion Matrix</li>
    </ul>

    <h3>🛠️ Technology Stack</h3>

    <ul>
        <li>Python</li>
        <li>Pandas</li>
        <li>NumPy</li>
        <li>Scikit-learn</li>
        <li>Plotly</li>
        <li>Streamlit</li>
        <li>Joblib</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:20px;">

    <h2>👨‍💻 About the Developer</h2>

    <h3>Anand Kumar</h3>

    <p>
        <b>Entry-Level Data Analyst</b>
    </p>

    <p>
        <b>Education:</b> B.A. Geography Honours | Magadh University | 2023
    </p>

    <p>
        <b>Training:</b> Data Analytics Training & Certification | iScale
    </p>

    <h3>💻 Technical Skills</h3>

    <p>
        Python • SQL • Excel • Pandas • NumPy • Power BI • Tableau •
        Matplotlib • Seaborn • Scikit-learn • Statistics • EDA •
        Data Cleaning • Power Query • DAX
    </p>

    <h3>📂 Projects</h3>

    <ul>
        <li>Student Career Success Prediction</li>
        <li>Movie Recommender System</li>
        <li>London Bike Ride Tableau Dashboard</li>
        <li>WhatsApp Chat Analyzer</li>
        <li>Music Store SQL Analysis</li>
        <li>Amazon Prime Video Analysis Dashboard</li>
        <li>Retail Sales Performance Dashboard</li>
    </ul>

    <h3>🔗 Profiles</h3>

    <p>
        <b>GitHub:</b>
        <a href="https://github.com/anandpal3244-coder" target="_blank">
            github.com/anandpal3244-coder
        </a>
    </p>

    <p>
        <b>LinkedIn:</b>
        <a href="https://www.linkedin.com/in/anand-pal-6a657b393/" target="_blank">
            linkedin.com/in/anand-pal-6a657b393
        </a>
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.info(
        "This application is intended for educational and "
        "analytical purposes. Model predictions should not be "
        "treated as guaranteed placement outcomes."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<br><br>

<div style="
    text-align:center;
    padding:25px;
    border-top:1px solid rgba(255,255,255,0.08);
    color:#64748b;
">

    <b>Student Career Success Predictor</b><br>

    Created by Anand Kumar • Data Analyst<br>

    Machine Learning • Data Analytics • Streamlit

</div>

""", unsafe_allow_html=True)


