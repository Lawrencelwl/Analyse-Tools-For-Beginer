import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

def extract_high_mi_features(data_path, target_column, high_mi_quantile=0.75):
    df = pd.read_csv(data_path, na_values=['?', 'NA'])
    
    # Check if target column exists in dataframe
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    
    # Delete any rows with NaN values
    df = df.dropna()

    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        if column != target_column:  # Do not encode the target column
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

    # Ensure the target column is encoded if it's a string
    if df[target_column].dtype == 'object' or df[target_column].dtype.name == 'category':
        target_encoder = LabelEncoder()
        df[target_column] = target_encoder.fit_transform(df[target_column])

    # Separate features and target variable for mutual information
    X = df.drop(columns=[target_column])
    y = df[target_column]  # No need to ensure encoding, handled above
    
    data_type = infer_data_type(data_path, target_column)
    print("data_type: ", data_type)
    # Calculate mutual information
    if data_type is 'Continuous':
        mi = mutual_info_regression(X, y, discrete_features='auto')
    else:
        mi = mutual_info_classif(X, y, discrete_features='auto')

    # Create a Series to view and sort the results
    mi_series = pd.Series(mi, index=X.columns)
    mi_series = mi_series.sort_values(ascending=False)

    # Define threshold for High MI Scores
    high_threshold = mi_series.quantile(high_mi_quantile)

    # Filter features based on the threshold
    high_mi_features = mi_series[mi_series >= high_threshold].index.tolist()

    return high_mi_features

def infer_data_type(csv_file_path, target_column):
    # Load the data
    data = pd.read_csv(csv_file_path)
    
    # Get the target column data
    column_data = data[target_column]

    if pd.api.types.is_numeric_dtype(column_data):
        data_type = 'Continuous'
    elif pd.api.types.is_string_dtype(column_data) or pd.api.types.is_categorical_dtype(column_data):
        data_type = 'Categorical'
    else:
        data_type = 'Unknown Data Type'

    return data_type