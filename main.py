import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Data Loading and Exploration
x_train_df = pd.read_csv('X_train.csv')
y_train_df = pd.read_csv('y_train.csv')
x_test_df = pd.read_csv('X_test.csv')

# Merge the X_train and y_train dataframes based on the Unique_ID column
train_df = pd.merge(x_train_df, y_train_df, on='Unique_ID')

# Drop the 'Unique_ID' column as it is not needed for training
train_df = train_df.drop('Unique_ID', axis=1)

# Convert categorical columns to numeric representation
categorical_columns = []
for i in range(8):
    categorical_columns.append(f'C{i + 1}')
for i in range(35):
    if i != 12:
        categorical_columns.append(f'N{i + 1}')
for col in categorical_columns:
    train_df[col] = train_df[col].astype('category').cat.codes

# Handle missing values in numerical columns
train_df = train_df.fillna(0)  # Replace missing values with 0

# Separate the features and target variable
X_train = train_df.drop('Dependent_Variable', axis=1)
y_train = train_df['Dependent_Variable']

# Train the model using the entire training data
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Preprocess the test data
x_test_processed = x_test_df.drop('Unique_ID', axis=1)

# Convert categorical columns to numeric representation
for col in categorical_columns:
    x_test_processed[col] = x_test_processed[col].astype('category').cat.codes

# Handle missing values in numerical columns
x_test_processed = x_test_processed.fillna(0) #fill missing values with 0
# Generate predictions for the test data
predictions = model.predict_proba(x_test_processed)[:, 1]

submission_df = pd.DataFrame({'Unique_ID': x_test_df['Unique_ID'], 'Class_1_Probability': predictions})
submission_df.to_csv('final_predictions.csv', index=False)
