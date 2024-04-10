import pandas as pd
from datawig.utils import random_split
from datawig import SimpleImputer
from checkRelationship import extract_high_mi_features
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse

def dataWig(data_path, output_col_data):
    df = pd.read_csv(data_path, na_values='?')

    # Perform train-test split (Default is 80/20 split)
    df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])
    
    output_col = output_col_data
    
    print("NaN count in", output_col ," before imputation:", df_train[output_col].isna().sum())

    # Define columns with useful info for the to-be-imputed column
    input_cols = extract_high_mi_features(df, target_column=output_col)

    # Initialize imputer for categorical imputation
    model_path = f'./Analyse Tools For Beginer/artifacts/imputer_model_{output_col}'
    imputer_cat = SimpleImputer(
        input_columns=input_cols,
        output_column=output_col,
        output_path=model_path
    )

    # Fit the imputer model on the training data
    imputer_cat.fit(train_df=df_train)

    # Impute missing values and return original dataframe with predictions
    predictions_cat = imputer_cat.predict(df_test)
    
    # # Evaluation of categorical imputation
    # from sklearn.metrics import matthews_corrcoef
    # df_test[output_col] = df_test[output_col].astype(str)
    # mcc_datawig = matthews_corrcoef(
    #     df_test[output_col],
    #     predictions_cat[f'{output_col}_imputed']
    # )
    
    # df_test[output_col] = df_test[output_col].astype(str)
    # # mse_datawig = (mse(predictions_cat[output_col], predictions_cat[f'{output_col}_imputed']))**0.5  
    # if pd.api.types.is_string_dtype(df_test[output_col]):
    #     mse_datawig = accuracy_score(df_test[output_col], predictions_cat[f'{output_col}_imputed'])
    #     print(f"The data in '{output_col}' is of string type.")
    # else:
    #     mse_datawig = (mse(predictions_cat[output_col], predictions_cat[f'{output_col}_imputed']))**0.5    
    #     print(f"The data in '{output_col}' is float.")
    
    # Convert both the true and imputed data to string if they are categorical
    df_test[output_col] = df_test[output_col].astype(str)
    predictions_cat[f'{output_col}_imputed'] = predictions_cat[f'{output_col}_imputed'].astype(str)

    # Use accuracy_score to compute the accuracy
    datawig_accuracy = accuracy_score(df_test[output_col], predictions_cat[f'{output_col}_imputed'])
    print(f"Accuracy for Datawig imputations: {datawig_accuracy}")
    
    print("NaN count in workclass after imputation:", predictions_cat[f'{output_col}_imputed'].isna().sum())

    # Return the model path and the MCC score
    return model_path, datawig_accuracy

# # Example usage:

# # Read the CSV file into a DataFrame
# data_path = pd.read_csv('./Analyse Tools For Beginer/data/csv_result-dataset_183_adult.csv', na_values='?')

# # Define a mapping for class labels to binary values, adjust if needed
# class_mapping = {'<=50K': 0, '>50K': 1}
# data_path['class'] = data_path['class'].map(class_mapping)

# # Check for NaN values in each column
# columns_with_nan = data_path.columns[data_path.isna().any()].tolist()
# print(columns_with_nan)

# # Loop over each column with NaN values and perform imputation and evaluation
# for output_col_cat in columns_with_nan:
#     model_path, mcc_score = dataWig(data_path, output_col_cat)
#     print(f"Model Path: {model_path}")
#     print(f"MCC Score: {mcc_score}")

# output_col_cat = 'workclass'
# model_path, mcc_score = dataWig(data_path, output_col_cat)
# print(f"Model Path: {model_path}")
# print(f"MCC Score: {mcc_score}")