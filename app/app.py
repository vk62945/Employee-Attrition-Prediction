import streamlit as st

st.set_page_config(
    page_title = "Employee Attrition Prediction",
    page_icon = "📊",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

st.title("📊 Employee Attrition Prediction System")

st.markdown("""
This application predicts whether an employee is likely to leave the company
using a Machine Learning model trained on the IBM HR Analytics dataset.

Fill in the employee details and click **Predict Attrition**.
""")

st.divider()

with st.sidebar:
    st.header("About")
    st.write("""
    **Project:** Employee Attrition Prediction
    **Model:** Logistic Regression
    **Framework:** Scikit-Learn Pipeline
    **Frontend:** Streamlit
    """)
    st.divider()
    st.info(
        "This project was built as part of an end-to-end Machine Learning portfolio."
    )

st.subheader("Employee Information")
st.write(
    "Please enter the employee details below"
)

col1, col2 = st.columns(2) 

with col1:
    st.write("Column 1")
with col2:
    st.write("Column 2")

st.divider()

st.caption(
    "Built using Python, Scikit-Learn and Streamlit"
)
