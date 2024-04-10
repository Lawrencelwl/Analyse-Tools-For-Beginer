import pandas as pd
from datawig.utils import random_split
from datawig import SimpleImputer
from checkRelationship import extract_high_mi_features
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt

def dataWig(data_path, output_col_data):
    df = pd.read_csv(data_path, na_values=['?', 'NA'])

    # Perform train-test split (Default is 80/20 split)
    df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])

    output_col = output_col_data

    # Define columns with useful info for the to-be-imputed column
    input_cols = extract_high_mi_features(data_path, target_column=output_col)

    # Initialize imputer for categorical imputation
    model_path = f'./Analyse Tools For Beginer/artifacts/datawig_model_{output_col}'
    imputer_cat = SimpleImputer(
        input_columns=input_cols,
        output_column=output_col,
        output_path=model_path
    )

    # Fit the imputer model on the training data
    imputer_cat.fit(train_df=df_train)

    # Impute missing values and return original dataframe with predictions
    predictions_cat = imputer_cat.predict(df_test)

    data_type = infer_data_type(data_path, output_col)

    # If the data is continuous, compute RMSE
    if data_type == 'Continuous':
        df_test[output_col] = pd.to_numeric(df_test[output_col], errors='coerce')
        predictions_cat[f'{output_col}_imputed'] = pd.to_numeric(predictions_cat[f'{output_col}_imputed'], errors='coerce')

        # Drop rows where either actual or imputed data is NaN before calculating RMSE
        mask = ~df_test[output_col].isna() & ~predictions_cat[f'{output_col}_imputed'].isna()
        rmse = sqrt(mean_squared_error(df_test[output_col][mask], predictions_cat[f'{output_col}_imputed'][mask]))
        metric = rmse
        print(f"RMSE for Datawig imputations: {rmse}")

    # If the data is categorical, compute accuracy
    elif data_type == 'Categorical':
        df_test[output_col] = df_test[output_col].astype(str)
        predictions_cat[f'{output_col}_imputed'] = predictions_cat[f'{output_col}_imputed'].astype(str)

        # Only calculate accuracy on non-NaN data
        mask = ~df_test[output_col].isna() & ~predictions_cat[f'{output_col}_imputed'].isna()
        accuracy = accuracy_score(df_test[output_col][mask], predictions_cat[f'{output_col}_imputed'][mask])
        metric = accuracy
        print(f"Accuracy for Datawig imputations: {accuracy}")

    return model_path, metric

def infer_data_type(csv_file_path, target_column):
    # Load the data
    data = pd.read_csv(csv_file_path)
    
    # Get the target column data
    column_data = data[target_column]

    # Determine the type based on the unique ratio and the data type
    if pd.api.types.is_numeric_dtype(column_data):
        data_type = 'Continuous'
    elif pd.api.types.is_string_dtype(column_data) or pd.api.types.is_categorical_dtype(column_data):
        # For string or categorical data, consider it categorical by default
        data_type = 'Categorical'
    else:
        data_type = 'Unknown Data Type'

    return data_type

