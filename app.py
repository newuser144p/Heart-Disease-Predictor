import streamlit as st 
import pandas as pd
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
    RestingBP = st.number_input("Resting Blood Pressure (mm/Hg)", min_value=0)
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
                -RestingECG: resting electrocardiographic results [0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy]\n
                -MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]\n
                -ExerciseAngina: exercise-induced angina [0: No, 1: Yes]\n
                -Oldpeak: oldpeak (ST depression)\n
                -ST_Slope: slope of the peak exercise ST segment [0: Upsloping, 1: Flat, 2: Downsloping]
        """)
    
    # Create a file uploader in the sidebar
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file into a DataFrame
        input_data = pd.read_csv(uploaded_file)
        
        try:
            model = pickle.load(open("LogisticRegression.pkl", "rb"))

            # Ensure that the input DataFrame matches the expected columns
            expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
                                'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

            if set(expected_columns).issubset(input_data.columns):
                
                # Step 1: Isolate only the required 11 columns to pass to the model
                features_for_prediction = input_data[expected_columns]

                # Step 2: Predict all rows at once! (No for loop)
                # This automatically creates a new column with the correct integer data type
                input_data['Prediction LR'] = model.predict(features_for_prediction)

                # Step 3: Save and display results
                input_data.to_csv('PredictedHeartLR.csv', index=False)

                st.subheader("Predictions:")
                st.write(input_data)

                # Create a button to download the updated CSV file
                st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
                
            else:
                st.warning("Please make sure the uploaded file has the correct columns.")
                
        except Exception as e:
            st.error(f"⚠️ Error: {e}. Please make sure 'LogisticRegression.pkl' is in your folder.")

    else:
        st.info("Upload a CSV file to get predictions.")

with tab3:
    import plotly.express as px
    st.title("Model Information")
    st.subheader("📊 Model Accuracy Comparison")
    
    # A much cleaner way to create the DataFrame directly
    data = {
        'Models': ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine'],
        'Accuracies': [80.97, 85.86, 84.23, 84.22]
    }
    df = pd.DataFrame(data)
    
    # Create the base chart with colors and text labels
    fig = px.bar(
        df, 
        x='Models', 
        y='Accuracies',
        color='Models', # Assigns a unique color to each bar
        text='Accuracies', # Tells Plotly to put the numbers on the bars
        color_discrete_sequence=px.colors.qualitative.Pastel # Uses a beautiful, modern color palette
    )

    # Fine-tune the aesthetics
    fig.update_traces(
        texttemplate='%{text:.2f}%', # Formats the labels as percentages (e.g., 85.86%)
        textposition='outside',      # Places the text neatly above the bars
        marker_line_color='black',   # Adds a subtle border to the bars
        marker_line_width=1,
        width=0.5                    # Makes the bars a bit thinner and more elegant
    )
    
    fig.update_layout(
        xaxis_title="",                  # Removes the redundant "Models" label on the X-axis
        yaxis_title="Accuracy (%)",      # Cleans up the Y-axis label
        yaxis=dict(range=[75, 90]),      # Zooms the Y-axis to highlight the differences!
        showlegend=False,                # Hides the legend (since the X-axis already has the names)
        plot_bgcolor='rgba(0,0,0,0)'     # Gives the chart a clean, transparent background
    )
    
    # use_container_width=True forces the chart to beautifully fill the screen space
    st.plotly_chart(fig, use_container_width=True)
