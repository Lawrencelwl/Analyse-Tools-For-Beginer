import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

def neuralNetworks(data_path, target_column):
    # Load the data
    data = pd.read_csv(data_path)

    # Replace '?','Na' with NaN (missing value marker)
    data = data.replace(['?', 'NA'], np.nan)
    
    # One-hot encode categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    data_type = infer_data_type(data_path, target_column)
    
    # Check if the target column is in the list of categorical columns
    if target_column in cat_cols:
        print("neuralNetworks_categorical")
        data_filled, accuracy = neuralNetworks_categorical(data_path, target_column)
    else:
        print("neuralNetworks_number")
        data_filled, accuracy = neuralNetworks_number(data_path, target_column, data_type)

    return data_filled, accuracy

def neuralNetworks_number(data_path, target_column, data_type):
    # Load the data
    data = pd.read_csv(data_path)
    original_data = data.copy()

    # Replace '?', 'NA' with NaN (missing value marker)
    data = data.replace(['?', 'NA'], np.nan)

    # One-hot encode categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    data_encoded = pd.get_dummies(data, columns=cat_cols, dummy_na=True)
    
    # Split data into sets with and without missing values
    missing = data_encoded[data_encoded[target_column].isnull()]
    not_missing = data_encoded.dropna()
    
    # Split not_missing set into input and output
    X = not_missing.drop(target_column, axis=1)
    y = not_missing[target_column]

    # Normalize the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Sequential model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, verbose=0)
    
    # Evaluate the model using the test set
    y_test_pred = model.predict(X_test)
    
    # Calculate the mean squared error and then take the square root to get RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Use the trained model to predict the missing values
    missing_values_input = missing.drop(target_column, axis=1)
    print(missing_values_input)
    missing_values_input = scaler.transform(missing_values_input)  # Use the same scaler
    predicted_values = model.predict(missing_values_input)
    
    # Fill the missing values with the predicted ones
    missing.loc[data_encoded[target_column].isnull(), target_column] = predicted_values
    
    # Combine the datasets
    data_filled = pd.concat([not_missing, missing], ignore_index=True)
    
    data_filled = one_hot_decode_number(data_filled, cat_cols)
    
    accuracy = rmse
    
    return data_filled, accuracy

def neuralNetworks_categorical(data_path, target_column):
    # Load the data
    data = pd.read_csv(data_path)
    original_data = data.copy()
    
    print("data_path: ", data_path)
    
    # Replace '?' with NaN (missing value marker)
    data = data.replace('?', np.nan)
    
    # Perform train-test split (Default is 80/20 split)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # One-hot encode categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    data_encoded = pd.get_dummies(data, columns=cat_cols, dummy_na=True)
    encoded_target_cols = [col for col in data_encoded.columns if col.startswith(target_column)]
    encoded_target_cols_nan = [col for col in data_encoded.columns if col.startswith(target_column) and '_nan' in col]
    target_nan_column = encoded_target_cols_nan[0]
    
    # Identify the indices where target_nan_column column is 1
    nan_indices = data_encoded[target_nan_column] == 1
    
    # Set the corresponding indices in all 'encoded_target_cols' to np.nan
    for col in encoded_target_cols:
        data_encoded.loc[nan_indices, col] = np.nan

    # Split data into sets with and without missing values
    missing = data_encoded[data_encoded[target_nan_column].isnull()]
    not_missing = data_encoded.dropna()

     # Split not_missing set into input and output
    X = not_missing.drop(encoded_target_cols, axis=1)
    y = not_missing[encoded_target_cols]

    # Normalize the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Sequential model
    model = Sequential()
    model.add(Dense(64, activation='tanh', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(len(encoded_target_cols)))

    # Compile the model
    model.compile(optimizer='RMSprop', loss='categorical_crossentropy')
    

    # Train the model
    model.fit(X_train, y_train, epochs=100, verbose=0)

    missing_values_input = missing.drop(encoded_target_cols, axis=1)
    missing_values_input = scaler.transform(missing_values_input)
    predicted_values = model.predict(missing_values_input)
    
    # Since model.predict() returns a 2D numpy array, need to flatten it to 1D if target is not one-hot encoded
    if len(encoded_target_cols) == 1:
        predicted_values = predicted_values.flatten()

    # Fill the missing values with the predicted ones
    for i, col in enumerate(encoded_target_cols):
        missing.loc[:, col] = predicted_values[:, i]
    
    # Combine the datasets
    data_filled = pd.concat([not_missing, missing], ignore_index=True)

    # If the target column was categorical, we need to reverse the one-hot encoding
    if data[target_column].dtype == 'object':
        data_filled = one_hot_decode(data_filled, original_data, [target_column])
        
    train_data_filled, test_data_filled = train_test_split(data_filled, test_size=0.2, random_state=42)
    
    test_data[target_column] = test_data[target_column].astype(str)
    test_data_filled[target_column] = test_data_filled[target_column].astype(str)
    test_data[f'{target_column}_imputed'] = test_data_filled[target_column]
    test_data[f'{target_column}_imputed'] = test_data[f'{target_column}_imputed'].astype(str)
    
    accuracy = accuracy_score(test_data[target_column], test_data_filled[target_column])

    return data_filled, accuracy

# Function to reverse one-hot encoding
def one_hot_decode(encoded_df, original_data, cols):
    for col in cols:
        category_cols = [c for c in encoded_df if c.startswith(col) and '_nan' not in c]
        
        # Create a custom function to handle NaN values and splitting the column name
        def handle_split(x):
            if isinstance(x, str):  # Check if 'x' is a string
                return x.split('_')[-1]
            return np.nan  # Return NaN if it's not a string (for example, if it's a NaN)
        
        # Apply the custom function to the DataFrame
        encoded_df[col] = encoded_df[category_cols].idxmax(axis=1).apply(handle_split)
        
        # Assign the decoded values back to the original DataFrame
        original_data[col] = encoded_df[col]
        
    return original_data

def one_hot_decode_number(encoded_df, original_cat_cols):
    # Initialize the DataFrame that will store the original categorical values
    decoded_df = pd.DataFrame(index=encoded_df.index)
    
    # Loop over the original categorical columns
    for col in original_cat_cols:
        # Find the one-hot encoded columns for the current categorical column
        encoded_cols = [c for c in encoded_df if c.startswith(f"{col}_")]
        # Get the subset of the DataFrame with the encoded columns
        sub_df = encoded_df[encoded_cols]
        # Find the original categorical value from the column names with max value
        decoded_df[col] = sub_df.idxmax(axis=1).str.replace(f"{col}_", "")
        # Replace the placeholder string for NaN values if it was added
        decoded_df[col] = decoded_df[col].replace("nan", np.nan)
    
    # Drop the one-hot encoded columns from the original DataFrame
    for col in original_cat_cols:
        encoded_cols = [c for c in encoded_df if c.startswith(f"{col}_")]
        encoded_df = encoded_df.drop(encoded_cols, axis=1)
    
    # Concatenate the original DataFrame with the decoded categorical values
    result_df = pd.concat([encoded_df, decoded_df], axis=1)
    
    return result_df

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
