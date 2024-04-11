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
    
    # Convert both the true and imputed data to string if they are categorical
    df_test[output_col] = df_test[output_col].astype(str)
    predictions_cat[f'{output_col}_imputed'] = predictions_cat[f'{output_col}_imputed'].astype(str)

    # Use accuracy_score to compute the accuracy
    datawig_accuracy = accuracy_score(df_test[output_col], predictions_cat[f'{output_col}_imputed'])
    print(f"Accuracy for Datawig imputations: {datawig_accuracy}")
    
    print("NaN count in workclass after imputation:", predictions_cat[f'{output_col}_imputed'].isna().sum())

    # Return the model path and the MCC score
    return model_path, datawig_accuracy
