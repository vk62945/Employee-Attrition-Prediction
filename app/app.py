from pathlib import Path
import sys
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.predict import predict_attrition
from src.predict import load_prediction_model

@st.cache_resource
def get_model():
    """
    Load the trained pipeline once and cache it.
    """
    return load_prediction_model()

st.set_page_config(
    page_title = "Employee Attrition Prediction",
    page_icon = "📊",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

st.title("📊 Employee Attrition Prediction System")
with st.expander("ℹ️ About this Project"):

    st.markdown("""
This application predicts whether an employee is likely to leave the company.

The prediction is based on employee-related information such as:

- Job Role
- Monthly Income
- Overtime
- Years at Company
- Job Satisfaction
- Work-Life Balance

The model was trained using the IBM HR Analytics Employee Attrition dataset.

This project demonstrates an end-to-end Machine Learning workflow including:

- Data preprocessing
- Feature engineering
- Model evaluation
- Production-ready pipeline
- Streamlit deployment
""")

st.divider()

with st.sidebar:
    st.header("📊 Project Information")
    st.markdown("""
    **Model**
    - Logistic Regression

    **Dataset**
    - IBM HR Analytics Employee Attrition

    **Framework**
    - Scikit-Learn Pipeline

    **Frontend**
    - Streamlit
    """)
    st.divider()
    st.header("📈 Model Performance")
    st.metric("Accuracy", "86.39%")
    st.metric("F1-Score", "44.44%")
    st.metric("ROC-AUC", "65.20%")
    st.divider()
    st.info(
        "This application predicts employee attrition using a machine learning model."
    )

st.subheader("Employee Information")
st.write(
    "Please enter the employee details below"
)

col1, col2 = st.columns(2) 

with col1:
    st.subheader("👤 Personal Information")
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value = 30
    )
    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )
    marital_status = st.selectbox(
        "Marital Status",
        [
            "Divorced",
            "Married",
            "Single"
        ]
    )
    education = st.selectbox(
        "Education",
        [1,2,3,4,5]
    )
    education_field = st.selectbox(
        "Education Field",
        [
            "Human Resources",
            "Life Sciences",
            "Marketing",
            "Medical",
            "Other",
            "Technical Degree"
        ]
    )

    st.subheader("💼 Job Information")
    business_travel = st.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )

    department = st.selectbox(
        "Department",
        [
            "Human Resources",
            "Research & Development",
            "Sales"
        ]
    )
    job_role = st.selectbox(
        "Job Role",
        [
            "Healthcare Representative",
            "Human Resources",
            "Laboratory Technician",
            "Manager",
            "Manufacturing Director",
            "Research Director",
            "Research Scientist",
            "Sales Executive",
            "Sales Representative"
        ]
    )
    job_level = st.selectbox(
        "Job Level",
        [1,2,3,4,5]
    )
    overtime = st.selectbox(
        "Over Time",
        [
            "No",
            "Yes"
        ]
    )

    st.subheader("📈 Satisfaction & Performance")
    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1,2,3,4]
    )
    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        [1,2,3,4]
    )
    relationship_satisfaction = st.selectbox(
        "Relationship Satisfaction",
        [1,2,3,4]
    )
    job_involvement = st.selectbox(
        "Job Involvement",
        [1,2,3,4]
    )
    performance_rating = st.selectbox(
        "Performance Rating",
        [3,4]
    )


with col2:
    st.subheader("💰 Salary & Work Details")
    daily_rate = st.number_input(
        "Daily Rate",
        min_value=100,
        max_value=1600,
        value=800
    )
    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=30,
        max_value=100,
        value=60
    )
    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=25000,
        value=5000
    )
    monthly_rate = st.number_input(
        "Monthly Rate",
        min_value=2000,
        max_value=30000,
        value=12000
    )
    percent_salary_hike = st.number_input(
        "Percent Salary Hike",
        min_value=10,
        max_value=30,
        value=15
    )

    st.subheader("📅 Experience")
    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=8
    )
    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=5
    )
    years_in_current_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        max_value=20,
        value=3
    )
    years_since_last_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=15,
        value=1
    )
    years_with_current_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=20,
        value=3
    )

    st.subheader("📋 Other Information")
    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=50,
        value=5
    )    
    training_times_last_year = st.selectbox(
        "Training Times Last Year",
        list(range(7))
    )
    work_life_balance = st.selectbox(
        "Work Life Balance",
        [1,2,3,4]
    )
    num_companies_worked = st.selectbox(
        "Companies Worked",
        list(range(10))
    )
    stock_option_level = st.selectbox(
        "Stock Option Level",
        [0,1,2,3]
    )

employee_data = {
    "Age": age,
    "BusinessTravel": business_travel,
    "DailyRate": daily_rate,
    "Department": department,
    "DistanceFromHome": distance_from_home,
    "Education": education,
    "EducationField": education_field,
    "EnvironmentSatisfaction": environment_satisfaction,
    "Gender": gender,
    "HourlyRate": hourly_rate,
    "JobInvolvement": job_involvement,
    "JobLevel": job_level,
    "JobRole": job_role,
    "JobSatisfaction": job_satisfaction,
    "MaritalStatus": marital_status,
    "MonthlyIncome": monthly_income,
    "MonthlyRate": monthly_rate,
    "NumCompaniesWorked": num_companies_worked,
    "OverTime": overtime,
    "PercentSalaryHike": percent_salary_hike,
    "PerformanceRating": performance_rating,
    "RelationshipSatisfaction": relationship_satisfaction,
    "StockOptionLevel": stock_option_level,
    "TotalWorkingYears": total_working_years,
    "TrainingTimesLastYear": training_times_last_year,
    "WorkLifeBalance": work_life_balance,
    "YearsAtCompany": years_at_company,
    "YearsInCurrentRole": years_in_current_role,
    "YearsSinceLastPromotion": years_since_last_promotion,
    "YearsWithCurrManager": years_with_current_manager
}
def validate_inputs():
    if years_at_company > total_working_years:
        return False, "Years at Company cannot exceed Total Working Years."

    if years_in_current_role > years_at_company:
        return False, "Years in Current Role cannot exceed Years at Company."

    if years_with_current_manager > years_at_company:
        return False, "Years with Current Manager cannot exceed Years at Company."

    return True, ""

is_valid, message = validate_inputs()
if not is_valid:
    st.error(message)
    st.stop()

model = get_model()

predict_button = st.button(
    "🔍 Predict Attrition",
    use_container_width=True
)

st.caption(
    "To reset the form, refresh the page or restore the default values."
)

if predict_button:
    with st.spinner("Predicting..."):
        result = predict_attrition(model, employee_data)

    prediction = result["prediction"]
    confidence = result["confidence"]

    st.divider()

    st.subheader("Prediction Result")
    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == "Yes":

            st.error(
                "⚠️ Employee is likely to leave the company."
            )
        else:
            st.success(
                "✅ Employee is likely to stay with the company."
            )
    with result_col2:
        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )
    st.progress(confidence / 100)
    st.subheader("Interpretation")
    if prediction == "Yes":

        st.warning(
            """
            The model predicts that this employee has a higher likelihood of leaving the company.

            HR may consider reviewing:

            • Career growth opportunities

            • Compensation

            • Work-life balance

            • Employee engagement
            """
        )

    else:

        st.info(
            """
            The model predicts that this employee is likely to stay with the company.

            Continue maintaining:

            • Positive work environment

            • Employee engagement

            • Career development
            """
        )
    with st.expander("Sumitted Employee Details"):
        st.json(employee_data)
    from datetime import datetime
    st.caption(
        f"Prediction generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:S')}"
    )
    # with st.expander("Raw Prediction Details"):
    #     st.json(result)

st.divider()

st.markdown("""
### 👨‍💻 Developer

**Vivek Kumar**

AWS Data Engineer | Aspiring AI Engineer

Built using:

- Python
- Scikit-Learn
- Streamlit
- Logistic Regression

Developed as an end-to-end Machine Learning project.

Machine Learning Portfolio Project
""")
