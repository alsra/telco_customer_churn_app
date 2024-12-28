import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, option, columns=None):
    """
    This function performs all necessary preprocessing steps on the churn dataframe.
    It includes feature selection, encoding of categorical variables, handling of missing data, 
    feature scaling, and data splitting to prepare the dataset for model training.
    """
    def binary_map(feature):
        valid_values = {'Yes': 1, 'No': 0}
        return feature.map(valid_values).fillna(-1)  # Handling missing values

    # Binary encoding for categorical features
    binary_list = ['SeniorCitizen', 'Dependents', 'PhoneService', 'PaperlessBilling']
    df[binary_list] = df[binary_list].apply(binary_map)

    # Encoding based on the option (Online or Batch)
    if option == "Online":
        # List of columns used for Online option
        columns = ['SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
                   'MultipleLines_No_phone_service', 'MultipleLines_Yes', 'InternetService_Fiber_optic', 'InternetService_No',
                   'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes', 'OnlineBackup_No_internet_service', 'TechSupport_No_internet_service',
                   'TechSupport_Yes', 'StreamingTV_No_internet_service', 'StreamingTV_Yes', 'StreamingMovies_No_internet_service', 'StreamingMovies_Yes',
                   'Contract_One_year', 'Contract_Two_year', 'PaymentMethod_Electronic_check']
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)
    elif option == "Batch":
        # List of columns used for Batch option
        columns = ['SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
                   'MultipleLines_No_phone_service', 'MultipleLines_Yes', 'InternetService_Fiber_optic', 'InternetService_No', 
                   'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes', 'OnlineBackup_No_internet_service', 'TechSupport_No_internet_service',
                   'TechSupport_Yes', 'StreamingTV_No_internet_service', 'StreamingTV_Yes', 'StreamingMovies_No_internet_service', 'StreamingMovies_Yes',
                   'Contract_One_year', 'Contract_Two_year', 'PaymentMethod_Electronic_check']
        df = df[['SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                 'OnlineBackup', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                 'MonthlyCharges', 'TotalCharges']]
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)
    else:
        raise ValueError("Invalid option. Please select 'Online' or 'Batch'.")

    # Fill missing values
    df['tenure'].fillna(df['tenure'].median(), inplace=True)
    df['MonthlyCharges'].fillna(df['MonthlyCharges'].median(), inplace=True)
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Feature scaling
    sc = MinMaxScaler()
    df['tenure'] = sc.fit_transform(df[['tenure']])
    df['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']])
    df['TotalCharges'] = sc.fit_transform(df[['TotalCharges']])

    # Return the processed dataframe along with columns if it's the training data
    return df
