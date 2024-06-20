import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the data
df = pd.read_csv('new_file.csv')
df = df.drop(columns=['index'], errors='ignore')

# Define features and target variable
X = df.drop(columns=['FirstName', 'LastName', 'CurrentDate', 'Doj', 'LeavesRemaining', 'Salary'])
y = df['Salary'].values  # Corrected target variable name to 'Salary'

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

# Define numerical and categorical features
numeric_features = X.select_dtypes(include=['int64', 'int32', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# Define transformations for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(max_depth=10, min_samples_leaf=4, n_estimators=10))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Save the pipeline to a file
with open("model.pkl", 'wb') as file:
    pickle.dump(pipeline, file)

# Print the shape of the test set
print(X_test.shape)
print(df.columns)
