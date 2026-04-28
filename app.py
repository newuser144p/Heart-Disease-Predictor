import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import base64

# Function to create a download link for a Daraframe as a CSV file
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions csv</a>'
    return href

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

with tab1:
    Age = st.number_input("Age", min_value=0, max_value=120 )
    Sex = st.selectbox("Sex", ["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Angina Pain", "Asymptomatic"])
    RestingBP = st.number_input("REsting Blood Pressure (mm/Hg)", min_value=0)
    Cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    FastingBS = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    RestingECG = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    MaxHR = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    ExerciseAngina = st.selectbox("Exercise_Induced Angina", ["Yes", "No"])
    Oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    ST_Slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical inputs to numerical values
    Sex = 0 if Sex == "Male" else 1
    ChestPainType = ["Typical Angina", "Atypical Angina", "Non-Angina Pain", "Asymptomatic"].index(ChestPainType)
    FastingBS = 1 if FastingBS == "> 120 mg/dl" else 0
    RestingECG = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(RestingECG)
    ExerciseAngina = 1 if ExerciseAngina == "Yes" else 0
    ST_Slope = ["Upsloping", "Flat", "Downsloping"].index(ST_Slope)

    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'Age': [Age],
        'Sex': [Sex],
        'ChestPainType': [ChestPainType],
        'RestingBP': [RestingBP],
        'Cholesterol': [Cholesterol],
        'FastingBS': [FastingBS],
        'RestingECG': [RestingECG],
        'MaxHR': [MaxHR],
        'ExerciseAngina': [ExerciseAngina],
        'Oldpeak': [Oldpeak],
        'ST_Slope': [ST_Slope]
    })

    algonames = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    modelnames = ['tree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl']

predictions = []
def predict_heart_disease(data):
    for modelname in modelnames:
        model = pickle.load(open(modelname, 'rb'))
        prediction =model.predict(data)
        predictions.append(prediction)
    return predictions

# Create a button to  make predictions
if st.button("Submit"):
    st.subheader('Results....')
    st.markdown('------------------------------------')

    result = predict_heart_disease(input_data)

    for i in range(len(predictions)):
        st.subheader(algonames[i])
        if result[i][0] == 0:
            st.write("No Heart Disease Detected")
        else:
            st.write("Heart Disease Detected")
        st.markdown('------------------------------------')

with tab2:
    st.title("Upload CSV File")
    st.subheader('Instructions to note before uploading the file:')
    st.info("""
            1. No NaN values allowed.
            2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 
            'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope')\n
            3. Check the spelling of the features names.\n

            4. Feature values conventions:\n
                -Age: age of the patient (years)\n
                -Sex: sex of the patient [0: Male, 1: Female]\n
                -ChestPainType: chest pain type [0: Typical Angina, 1: Atypical Angina, 2: Non-Angina Pain, 3: Asymptomatic]\n
                -RestingBP: resting blood pressure (mm/Hg)\n
                -Cholesterol: serum cholesterol (mm/dl)\n
                -FastingBS: fasting blood sugar [0:if blood sugar <= 120 mg/dl, 1:if blood sugar > 120 mg/dl]\n
                -RestingECG: resting electrocardiographic results [0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy]
                -MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]\n
                -ExerciseAngina: exercise-induced angina [0: No, 1: Yes]\n
                -Oldpeak: oldpeak (ST depression)\n
                -ST_Slope: slope of the peak exercise ST segment [0: Upsloping, 1: Flat, 2: Downsloping]
            
            
            
        """)
    # Create a file uploader in the sidebar
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
# Read the iploaded CSV file= into a DataFrame
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open("LogisticRegression.pkl","rb"))

        # Ensure that the input Dataframe matches the expected columns and format
        expected_columns =['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
                           'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        if set(expected_columns).issubset(input_data.columns):
            input_data['Prediction LR'] = ''

            for i in range(len(input_data)):
                arr = input_data.iloc[i,:-1].values
                input_data['Prediction LR'][i] = model.predict([arr])[0]
            input_data.to_csv('PredictedHeartLR.csv')

            # Display the predictions
            st.subheader("Predictions:")
            st.write(input_data)

            # Create a button to download the updated CSV file
            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("Please make sure the uploaded file has the correct columns.")

    else:
        st.info("Upload a CSV file to get predictions.")

with tab3:
    import plotly.express as px
    data = {'Decision Trees': 80.97, 'Logistic Regression': 85.86, 'Random Forest': 84.23, 'Support Vector Machine':84.22}
    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame (list(zip(Models,Accuracies)),columns=['Models','Accuracies'])
    fig = px.bar(df, y='Accuracies',x='Models')
    st.plotly_chart(fig)