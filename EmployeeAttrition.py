import pickle
import pandas as pd
import streamlit as st
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# Load the saved model
loaded_model = pickle.load(open(r'C:\framework\env\Scripts\random_forest_classifier_model.pkl', 'rb'))
loaded_model_attrition = pickle.load(open(r'C:\framework\env\Scripts\random_forest_model.pkl', 'rb'))
st.set_page_config(page_title="Employee Predictions", page_icon=":bar_chart:")
st.title("Employee Predictions :chart_with_upwards_trend:")
page = st.sidebar.selectbox("Select a Prediction", ("Attrition", "Performance Rating","Employee attrition risk"))

if page == "Attrition":
    st.header("Attrition Prediction :bust_in_silhouette:")
# Input features
    df=pd.read_csv(r"C:\Users\Meyyappan\Desktop\Employee-Attrition - Employee-Attrition.csv")
    le = preprocessing.LabelEncoder()
    job_role = st.selectbox("Job Role", df['JobRole'].unique()) # Example: Use a selectbox

# Encode the job role
    encoded_job_role = le.fit_transform([job_role])[0]
    job_role = st.number_input('Job Role', min_value=0, max_value=8, value=0)
    job_satisfaction = st.slider('Job Satisfaction', 1, 4, 1)
    age = st.number_input('Age', min_value=18, max_value=60, value=30)
    le_gender = LabelEncoder()
    le_gender.fit(df['Gender'])
    gender = st.radio("Gender", ('Male', 'Female'))
    
    # Encode gender using the fitted LabelEncoder
    encoded_gender = le_gender.transform([gender])[0]
    relationship_satisfaction = st.slider('Relationship Satisfaction', 1, 4, 1)
    years_at_company = st.number_input('Years At Company', min_value=0, max_value=40, value=0)
    tenure_category = st.selectbox('Tenure Category', [0,1,2,3], index=0) # Short term, Midterm, Longterm, VeryLongterm
    department_options = sorted(df['Department'].unique())  # Get unique department values from your dataframe
    department_index = st.selectbox("Department", department_options)
    # Convert department name to its encoded value using the fitted LabelEncoder
    encoded_department = le.fit_transform([department_index])[0] # Sales, Research & Development, Human Resources
    monthly_income = st.number_input('Monthly Income', min_value=1000, max_value=20000, value=5000)
    input_df = pd.DataFrame({'JobRole': [encoded_job_role], 'JobSatisfaction': [job_satisfaction], 'Age': [age], 'Gender': [encoded_gender], 'RelationshipSatisfaction': [relationship_satisfaction], 'YearsAtCompany': [years_at_company], 'tenurecategory': [tenure_category], 'Department': [encoded_department], 'MonthlyIncome': [monthly_income]})
    if st.button("Predict Attrition"):
        prediction = loaded_model.predict (input_df)
        if prediction[0] == 0:
            st.write("Prediction: No :heavy_minus_sign:") # Display "No" if the prediction is 0
        else:
            st.write("Prediction: Yes :heavy_plus_sign:") # Display "Yes" if the prediction is 1
            
elif page =="Employee attrition risk":
    st.header("Employee attrition risk :red_circle:")
    df=pd.read_csv(r"C:\Users\Meyyappan\Desktop\Employee-Attrition - Employee-Attrition.csv")
    df = df.drop(['EmployeeCount','Over18', 'StandardHours'], axis=1)
    le = preprocessing.LabelEncoder()
    cols = ['Attrition','BusinessTravel','Department','EducationField', 'Gender','JobRole','MaritalStatus','OverTime']
    df[cols] = df[cols].apply(le.fit_transform)

    # Calculate Performance Score (as in the previous code)
    df['PerformanceScore'] = (
        0.3 * df['JobSatisfaction'] +
        0.2 * df['YearsAtCompany'] +
        0.3 * df['TrainingTimesLastYear'] +
        0.2 * (df['MonthlyIncome'] / df['MonthlyIncome'].max())
    )

    attrition_threshold = df['PerformanceScore'].median() # Example threshold

    df['AttritionRisk'] = df['PerformanceScore'].apply(lambda x: 'High' if x < attrition_threshold else 'Low')

    high_risk_df = df[df['AttritionRisk'] == 'High']
    low_risk_df = df[df['AttritionRisk'] == 'Low']

    # Display tables
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("High Attrition Risk Employees :rotating_light:")
        st.write(high_risk_df[['EmployeeNumber', 'AttritionRisk', 'PerformanceScore']])  # Display only needed columns

    with col2:
        st.subheader("Low Attrition Risk Employees :large_green_circle:")
        st.write(low_risk_df[['EmployeeNumber', 'AttritionRisk', 'PerformanceScore']])


elif page == "Performance Rating":
    st.header("Performance Rating Prediction :bar_chart:")
    df=pd.read_csv(r"C:\Users\Meyyappan\Desktop\Employee-Attrition - Employee-Attrition.csv")
    le = preprocessing.LabelEncoder()
    age = st.number_input("Age", min_value=18, max_value=100)
    monthly_income = st.number_input("Monthly Income",min_value=2000, value=5000, step=100)
    education = st.selectbox("Education", [1, 2, 3, 4, 5])
    job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
    job_level = st.slider("Job Level", min_value=1, max_value=5)
    years_in_current_role = st.slider("Years in Current Role",min_value=0, max_value=40)
    department_options = sorted(df['Department'].unique())  # Get unique department values from your dataframe
    department_index = st.selectbox("Department", department_options)
    # Convert department name to its encoded value using the fitted LabelEncoder
    encoded_department = le.fit_transform([department_index])[0]
    job_satisfaction = st.number_input("Job Satisfaction",min_value=1, max_value=4)
    tenure_category = st.selectbox("Tenure Category", [0,1,2,3])
    engagement_score = st.number_input("Engagement Score", min_value=0, max_value=100, value=50)
    st.write(f"Engagement Score (Percentage): {engagement_score}%")
    le_gender = LabelEncoder()
    le_gender.fit(df['Gender'])
    gender = st.radio("Gender", ('Male', 'Female'))
    
    # Encode gender using the fitted LabelEncoder
    encoded_gender = le_gender.transform([gender])[0]
    years_at_company = st.number_input("Years at Company",min_value=0, max_value=40)
    
    input_df = pd.DataFrame({
        'Age': [age], 'MonthlyIncome': [monthly_income], 'Education': [education],
        'JobInvolvement': [job_involvement], 'JobLevel': [job_level],
        'YearsInCurrentRole': [years_in_current_role], 'Department': [encoded_department],
        'JobSatisfaction': [job_satisfaction], 'tenurecategory': [tenure_category],
        'EngagementScore': [engagement_score], 'Gender': [encoded_gender],
        'YearsAtCompany': [years_at_company]
    })

# Display results
    if st.button("Predict Performance Rating"):
        prediction = loaded_model_attrition.predict(input_df)
        st.write(f"Performance Rating: {prediction[0]}")
        