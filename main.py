import os
import streamlit as st
import pandas as pd
from PIL import Image
import tensorflow as tf
from preprocessing import preprocess

# Load the Keras model from disk

model = tf.keras.models.load_model("Models/ann_model.keras")





def main():
    # Setting Application title
    st.title('Telco Customer Churn Prediction App')

    # Setting Application description
    st.markdown("""
     :iphone:  This Streamlit application is designed to predict customer churn in a fictional telecommunication use case.
     It offers a comprehensive solution for both individual and batch predictions, enabling users to analyze churn dynamics effectively.\n
        :bulb: **How to use this app?**\n
        - Online Predictions: Input customer details to obtain immediate insights into churn likelihood.\n
        - Batch Processing: Upload datasets for bulk analysis, facilitating predictions for multiple customers simultaneously.
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    # Sidebar configuration
    image = Image.open('app.jpg')
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        
        # Demographic data
        seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
        dependents = st.selectbox('Dependent:', ('Yes', 'No'))
        
        # Payment data
        tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
        PaymentMethod = st.selectbox('PaymentMethod', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
        monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
        totalcharges = st.number_input('The total amount charged to the customer', min_value=0, max_value=10000, value=0)

        # Services signed up for
        mutliplelines = st.selectbox("Does the customer have multiple lines", ('Yes', 'No', 'No phone service'))
        phoneservice = st.selectbox('Phone Service:', ('Yes', 'No'))
        internetservice = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
        onlinesecurity = st.selectbox("Does the customer have online security", ('Yes', 'No', 'No internet service'))
        onlinebackup = st.selectbox("Does the customer have online backup", ('Yes', 'No', 'No internet service'))
        techsupport = st.selectbox("Does the customer have technology support", ('Yes', 'No', 'No internet service'))
        streamingtv = st.selectbox("Does the customer stream TV", ('Yes', 'No', 'No internet service'))
        streamingmovies = st.selectbox("Does the customer stream movies", ('Yes', 'No', 'No internet service'))

        data = {
                'SeniorCitizen': seniorcitizen,
                'Dependents': dependents,
                'tenure':tenure,
                'PhoneService': phoneservice,
                'MultipleLines': mutliplelines,
                'InternetService': internetservice,
                'OnlineSecurity': onlinesecurity,
                'OnlineBackup': onlinebackup,
                'TechSupport': techsupport,
                'StreamingTV': streamingtv,
                'StreamingMovies': streamingmovies,
                'Contract': contract,
                'PaperlessBilling': paperlessbilling,
                'PaymentMethod': PaymentMethod,
                'MonthlyCharges': monthlycharges,
                'TotalCharges': totalcharges
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        # Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online')

        # Make prediction using the Keras model
        if st.button('Predict'):
            prediction = model.predict(preprocess_df)
            # Assuming the model returns probabilities, convert them to class predictions
            if prediction >= 0.5:  # Assuming 1 is churn, and the output is a probability
                st.warning('Yes, the customer will terminate the service.')
            else:
                st.success('No, the customer is happy with Telco Services.')

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            # Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                # Get batch prediction
                predictions = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(predictions, columns=["Predictions"])
                prediction_df = prediction_df.replace({1: 'Yes, the customer will terminate the service.',
                                                        0: 'No, the customer is happy with Telco Services.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)

if __name__ == '__main__':
    main()
