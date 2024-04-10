import re
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from dataWig import dataWig
from neuralNetworks import neuralNetworks
from datawig import SimpleImputer
from PyQt5.QtCore import QObject, pyqtSignal

class HandlerWorker(QObject):
    finished = pyqtSignal()
    data_ready = pyqtSignal(object)
    # data_ready = pyqtSignal(object, object, object, float)
    
    def setup_data(self, csv_file_path, selected_target, selected_data_columns, selected_model):
        self.csv_file_path = csv_file_path
        self.selected_target = selected_target
        self.selected_data_columns = selected_data_columns
        self.selected_model = selected_model
    
    def handle_data(self):
        selected_model = self.selected_model
        csv_file_path = self.csv_file_path
        selected_target = self.selected_target
        selected_data_columns = self.selected_data_columns
        
        data_type = infer_data_type(csv_file_path, selected_target)

        data_path = csv_file_path
        # Extract the CSV file name
        csv_file_name = os.path.basename(data_path)
        df = pd.read_csv(data_path, na_values=['?', 'NA'])
        
        selected_data_columns = [col for col in selected_data_columns if col in df.columns]
            
        check_columns_nan = selected_data_columns
            
        if selected_target in df.columns and selected_target not in selected_data_columns:
            check_columns_nan.append(selected_target)

        # Check for NaN values only in the specified columns
        columns_with_nan = [col for col in check_columns_nan if df[col].isna().any()]
        print(columns_with_nan)
            
        # If there are no NaN values in the specified columns, call the analyseData function
        if not columns_with_nan:
            if selected_model == "Logistic Regression Model":
                accuracy, fig = logisticRegression(csv_file_path, selected_data_columns, selected_target)
            else:
                accuracy, fig = randomForestClassifier(csv_file_path, selected_data_columns, selected_target)
            
        # Initialize an empty dictionary instead of a list
        columns_to_impute = {}
        accuracy = {}
        analyze_csv_path = {}
            
        # model directory
        model_directory = './Analyse Tools For Beginer/artifacts'
        exist_array = []

        # List all model files in the model directory
        try:
            model_in_directory = os.listdir(model_directory)
            model_file_paths = [os.path.join(model_directory, file_name) for file_name in model_in_directory]
            # Replace backslashes with forward slashes
            model_file_paths = [path.replace('\\', '/') for path in model_file_paths]
        except FileNotFoundError:
            print(f"The directory {model_directory} was not found.")
            model_in_directory = []
            
        #checking has model
        for partial_filename in columns_with_nan:
            found = any(partial_filename in file_name for file_name in model_in_directory)
            exist_array.append(found)
            
        if columns_with_nan:
            # Loop over each column with NaN values and perform imputation and evaluation
            # DataWig
            for index, output_col_cat in enumerate(columns_with_nan):
                if not exist_array[index]:
                    model_path, accuracy_datawig = dataWig(data_path, output_col_cat)
                    # Add the column name and its model path to the dictionary
                    columns_to_impute[output_col_cat] = model_path
                    accuracy[f"{output_col_cat}_dataWig"] = accuracy_datawig
                
            for output_col_cat in columns_with_nan:
                output_col = output_col_cat
                model_path = f'./Analyse Tools For Beginer/artifacts/datawig_model_{output_col}'
                
            # Neural Networks    
            neuralNetworks_temp = data_path
            intermediate_csv_path = './Analyse Tools For Beginer/data/temp_neuralNetworks.csv'
            final_csv_path = f'./Analyse Tools For Beginer/data/{csv_file_name[:-4]}_neuralNetworks.csv'
                
            for index, output_col_cat in enumerate(columns_with_nan):
                # Apply the sequential function to the current target column
                filled_data, neuralNetworks_accuracy = neuralNetworks(neuralNetworks_temp, output_col_cat)
                if index is len(columns_with_nan) - 1:
                    print("Fin: ",index)
                    # This is the last target column, save to the final CSV file
                    filled_data.to_csv(final_csv_path, index=False)
                    if data_type == 'Continuous':
                        convert_large_numbers_to_scientific_notation(final_csv_path)
                    filled_data = filled_data.dropna()
                    analyze_csv_path[f'{csv_file_name[:-4]}_neuralNetworks'] = final_csv_path
                    if os.path.exists(intermediate_csv_path):
                        os.remove(intermediate_csv_path)
                if index < len(columns_with_nan) - 1:
                    filled_data.to_csv(intermediate_csv_path, index=False)
                    # Update the csv_file_path to use the intermediate file for the next iteration
                    neuralNetworks_temp = intermediate_csv_path
                    if data_type == 'Continuous':
                        convert_large_numbers_to_scientific_notation(neuralNetworks_temp)
                accuracy[f"{output_col_cat}_neuralNetworks"] = neuralNetworks_accuracy

            # List all model files in the model directory
            try:
                model_in_directory = os.listdir(model_directory)
                model_file_paths = [os.path.join(model_directory, file_name) for file_name in model_in_directory]
                # Replace backslashes with forward slashes
                model_file_paths = [path.replace('\\', '/') for path in model_file_paths]
            except FileNotFoundError:
                model_in_directory = []
            
            # Impute each column using the pre-trained models
            for model_name, file_path in zip(model_in_directory, model_file_paths):
                column_name = model_name.replace('datawig_model_', '')
                columns_to_impute[column_name] = file_path

            for column_name in columns_with_nan:
                if column_name in columns_to_impute:
                    model_path = columns_to_impute[column_name]
                    df = replaceNan(df, column_name, model_path)
                    if f"{column_name}_dataWig" not in accuracy:
                        accuracy_for_datawig = get_accuracy_from_log(model_path)
                        accuracy.update(accuracy_for_datawig)
                    # print('accuracy_datawig_updata: ', accuracy)

            # Check if there are any NaN values in the specified columns of the dataframe
            if df[check_columns_nan].isnull().any().any():
                # If there are NaN values in the specified columns, handle the remaining cases
                print("There are still NaN values in the specified columns of the dataframe. Please check the imputation models and data.")
            else:
                # If there are no NaN values in the specified columns, proceed with saving the CSV
                # Define the path for the new CSV file
                new_csv_file_path = f'./Analyse Tools For Beginer/data/{csv_file_name[:-4]}_dataWig.csv'
                # Save the DataFrame to the new CSV file, ensuring not to include the index
                df.to_csv(new_csv_file_path, index=False)
                analyze_csv_path[f'{csv_file_name[:-4]}_dataWig'] = new_csv_file_path
                print("CSV file without NaN values in the specified columns has been saved.")
                
            # Run the comparison function
            comparison_results = compare_accuracy(accuracy, csv_file_path)
                
            new_csv_path = f'./Analyse Tools For Beginer/data/{csv_file_name[:-4]}_best_accuracy.csv'
            # Call the function with the provided dictionaries and the desired output CSV path
            file_name = f"{csv_file_name[:-4]}"
            create_combined_csv(comparison_results, analyze_csv_path, new_csv_path, file_name)
            if selected_model == "Logistic Regression Model":
                accuracy, fig = logisticRegression(csv_file_path, selected_data_columns, selected_target)
            else:
                accuracy, fig = randomForestClassifier(csv_file_path, selected_data_columns, selected_target)   

        # Emit finished signal when done
        self.data_ready.emit(fig)  # Emit the data_ready signal with the fig object
        self.finished.emit()

def check_missing_data(data_path):
    """
    Check if the CSV file at data_path has any missing data.

    :param data_path: Path to the CSV file
    :return: True if there are missing values, False otherwise
    """
    try:
        df = pd.read_csv(data_path, na_values=['?', 'NA'])
        # Check if there's any missing value in DataFrame
        if df.isnull().values.any():
            return True
        else:
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return True  # Assuming that an inability to read the file implies missing data

def replaceNan(df, column_name, model_path):
    """
    This function imputes missing values for a single column using a pre-trained SimpleImputer model.
    """
    # Load pre-trained imputer model
    imputer = SimpleImputer.load(model_path)
    
    # Impute missing values
    imputed = imputer.predict(df)
    
    # Replace the column in the original dataframe with the imputed values
    df[column_name] = imputed[f'{column_name}_imputed']
    
    return df

def convert_large_numbers_to_scientific_notation(csv_path):
    df = pd.read_csv(csv_path)
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].apply(lambda x: '{:.2e}'.format(x) if pd.notnull(x) else x)
    
    df.to_csv(csv_path, index=False)

def get_accuracy_from_log(model_path):
    # Compile the regular expression pattern to find accuracy
    accuracy_pattern = re.compile(r"Train-(.*)-accuracy=([0-9.]+)")
    
    # Initialize variable to hold the last accuracy value
    last_accuracy = None
    
    # Extract the category from the model path name
    category = os.path.basename(model_path).replace('datawig_model_', '')
    
    # Construct the correct file path for the imputer.log file
    log_file_path = os.path.join(model_path, 'imputer.log')
    
    # Open and read the log file
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                # Search for the accuracy pattern in each line
                match = accuracy_pattern.search(line)
                if match:
                    # Update the last_accuracy with the current value found
                    last_accuracy = float(match.group(2))
                    
    except FileNotFoundError:
        print(f"Log file not found for {model_path}.")
    
    # Return the category and the last accuracy as a dictionary
    return {f"{category}_dataWig": last_accuracy}

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

def compare_accuracy(accuracy_scores, csv_file_path):
    # Identify all unique categories based on the accuracy_scores keys
    categories = set(key.rsplit('_', 1)[0] for key in accuracy_scores.keys())

    # Initialize a dictionary to hold the comparison results
    comparison_results = {}

    # Compare the metric for each category
    for category in categories:
        datawig_key = f"{category}_dataWig"
        neural_networks_key = f"{category}_neuralNetworks"

        # Ensure both methods have scores for the category
        if datawig_key in accuracy_scores and neural_networks_key in accuracy_scores:
            datawig_score = accuracy_scores[datawig_key]
            neural_networks_score = accuracy_scores[neural_networks_key]

            # Use the infer_data_type function to determine if the data is Continuous or Categorical
            data_type = infer_data_type(csv_file_path, category)

            # Compare based on data type
            if data_type == 'Categorical':
                # For categorical data, higher is better (accuracy)
                better_method = 'dataWig' if datawig_score > neural_networks_score else 'neuralNetworks'
            elif data_type == 'Continuous':
                # For continuous data, lower is better (RMSE)
                better_method = 'dataWig' if datawig_score < neural_networks_score else 'neuralNetworks'
            else:
                print(f"Unknown data type for category: {category}")
                continue

            # Output the comparison results
            comparison_results[category] = {
                'DataWig': datawig_score,
                'Neural Networks': neural_networks_score,
                'Better Method': better_method,
                'Data Type': data_type
            }
        else:
            print(f"Missing scores for category: {category}")

    return comparison_results

def create_combined_csv(comparison_results, analyze_csv_path, output_csv_path, file_name):
    # Load the original data (assuming the original data is in one of the CSVs listed in analyze_csv_path)
    original_data_path = next(iter(analyze_csv_path.values()))  # Getting one CSV path to load original data
    original_data = pd.read_csv(original_data_path)

    # Create a new DataFrame to store the best columns
    best_columns_data = pd.DataFrame()

    # Iterate through comparison_results to get the highest accuracy for each column
    for column, methods in comparison_results.items():
        # Determine the best method based on the highest accuracy
        best_method = 'dataWig' if methods['Better Method'] is 'DataWig' else 'neuralNetworks'
        # Select the CSV file based on the best method determined
        source_file_key = f'{file_name}_{best_method}'
        source_file = analyze_csv_path[source_file_key]
        # Load the data from the source file
        source_data = pd.read_csv(source_file)
        # Add the best column to the new DataFrame
        best_columns_data[column] = source_data[column]

    # Optionally, add other columns from the original data that were not imputed
    for column in original_data.columns:
        if column not in best_columns_data.columns:
            best_columns_data[column] = original_data[column]

    # Save the new DataFrame as a CSV file
    best_columns_data.to_csv(output_csv_path, index=False)

def logisticRegression(csv_file_path, selected_data_columns, selected_target):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Map the target class to binary format
    label_encoder = LabelEncoder()
    df[selected_target] = label_encoder.fit_transform(df[selected_target])
    
    # Get the categorical data and one-hot encode it
    categorical_data = df[selected_data_columns]
    column_transformer = ColumnTransformer([('ohe', OneHotEncoder(), selected_data_columns)],
                                           remainder='passthrough')
    transformed_data = column_transformer.fit_transform(categorical_data)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(transformed_data, df[selected_target], test_size=0.2, random_state=42)
    
    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    
    # Map numeric predictions back to original labels for plotting
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Create a figure for the bar chart or pie chart
    fig, ax = plt.subplots()
    
    # Plot a bar chart or a pie chart depending on the number of unique values
    if len(set(y_pred_labels)) > 10:  # If target predictions have more than 10 unique values, use a bar chart
        value_counts = pd.Series(y_pred_labels).value_counts()
        ax.bar(value_counts.index.map(str), value_counts.values)  # Convert index to string if they are not
        ax.set_ylabel('Counts')
        ax.set_xlabel('Classes')
    else:  # Otherwise, use a pie chart
        ax.pie(pd.Series(y_pred_labels).value_counts(), labels=pd.Series(y_pred_labels).value_counts().index.map(str), autopct='%1.1f%%')
    
    # Remove the title
    ax.set_title("Logistic Regression Model")

    # Return the accuracy and the figure object
    return accuracy, fig

def randomForestClassifier(csv_file_path, selected_data_columns, selected_target):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Map the target class to binary format
    label_encoder = LabelEncoder()
    df[selected_target] = label_encoder.fit_transform(df[selected_target])
    
    # Get the categorical data and one-hot encode it
    categorical_data = df[selected_data_columns]
    column_transformer = ColumnTransformer([('ohe', OneHotEncoder(), selected_data_columns)],
                                           remainder='passthrough')
    transformed_data = column_transformer.fit_transform(categorical_data)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(transformed_data, df[selected_target], test_size=0.2, random_state=42)
    
    # Train a RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    
    # Map numeric predictions back to original labels for plotting
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Create a figure for the bar chart or pie chart
    fig, ax = plt.subplots()
    
    # Plot a bar chart or a pie chart depending on the number of unique values
    if len(set(y_pred_labels)) > 10:  # If target predictions have more than 10 unique values, use a bar chart
        value_counts = pd.Series(y_pred_labels).value_counts()
        ax.bar(value_counts.index.map(str), value_counts.values)  # Convert index to string if they are not
        ax.set_ylabel('Counts')
        ax.set_xlabel('Classes')
    else:  # Otherwise, use a pie chart
        ax.pie(pd.Series(y_pred_labels).value_counts(), labels=pd.Series(y_pred_labels).value_counts().index.map(str), autopct='%1.1f%%')
    
    # Remove the title
    ax.set_title("Random Forest Model")

    # Return the accuracy and the figure object
    return accuracy, fig

