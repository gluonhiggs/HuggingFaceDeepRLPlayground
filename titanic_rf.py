import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from collections import Counter
# Set pandas display options
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

# Set random seed for reproducibility
np.random.seed(42)

# Load the Titanic dataset (assuming standard Kaggle paths)
train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')

# Quick look at the data
# Print top 100 rows
print(train_df.head(100))
print(train_df.info())

# Check missing values
print(train_df.isnull().sum())
def detect_outlier(df, n, cols):
    outlier_indices = []
    for i in cols:
        Q1, Q3 = np.percentile(df[i].dropna(), [25, 75])  # Drop NaN for percentile
        IQR = Q3 - Q1
        step = 1.5 * IQR
        outlier_index_list = df[(df[i] < Q1 - step) | (df[i] > Q3 + step)].index
        outlier_indices.extend(outlier_index_list)
    outlier_indices = Counter(outlier_indices)
    return [k for k, v in outlier_indices.items() if v > n]

outliers = detect_outlier(train_df, 3, ['Age', 'SibSp', 'Parch', 'Fare'])
train = train_df.drop(outliers, axis=0).reset_index(drop=True)
# Function to preprocess data (train and test)
def preprocess_data(df:pd.DataFrame):
    # Copy to avoid modifying original
    df_processed = df.copy()
    
    # Fill missing values
    df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
    df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
    df_processed['Fare'].fillna(df_processed['Fare'].mode()[0], inplace=True)
    
    df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles and standardize others
    df_processed['Title'] = df_processed['Title'].replace(['Countess', 'Sir'], 'Rare')
    df_processed['Title'] = df_processed['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df_processed['Title'] = df_processed['Title'].replace(['Mme','Lady'], 'Mrs')
    # Extract Ticket Prefix
    def extract_ticket_prefix(ticket):
        # Split on space, take first part if it contains letters
        parts = ticket.split()
        prefix = parts[0] if len(parts) > 0 else ticket
        # If prefix is purely numeric, label it as 'NUMERIC'
        if prefix.replace('.', '').replace('/', '').isdigit():
            return 'NUMERIC'
        # Clean up prefix (remove trailing numbers if mixed)
        prefix = ''.join(c for c in prefix if not c.isdigit())
        return prefix.strip('.').strip('/')
    df_processed['TicketPrefix'] = df_processed['Ticket'].apply(extract_ticket_prefix)
    prefix_counts = df_processed['TicketPrefix'].value_counts()
    rare_prefixes = prefix_counts[prefix_counts < 10].index
    df_processed['TicketPrefix'] = df_processed['TicketPrefix'].replace(rare_prefixes, 'RARE')
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch']
    df_processed['FamilySizeCat'] = pd.cut(df_processed['FamilySize'], bins=[-1, 0, 3, 10], labels=[0, 1, 2])
    df_processed['FamilySizeCat'] = df_processed['FamilySizeCat'].astype(int)
    # Drop columns unlikely to help or too sparse
    df_processed.drop(['Cabin', 'Ticket', 'Name', 'FamilySize'], axis=1, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    df_processed['Sex'] = le.fit_transform(df_processed['Sex'])
    df_processed['Embarked'] = le.fit_transform(df_processed['Embarked'])
    df_processed['Title'] = le.fit_transform(df_processed['Title'])
    df_processed['TicketPrefix'] = le.fit_transform(df_processed['TicketPrefix'])
    df_processed['Pclass'] = le.fit_transform(df_processed['Pclass'])
    
    return df_processed

# Apply preprocessing
train_processed = preprocess_data(train_df)
test_processed = preprocess_data(test_df)
# Define features and target
X = train_processed.drop(['Survived', 'PassengerId'], axis=1)

y = train_processed['Survived']

X_test = test_processed.drop('PassengerId', axis=1)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Define features and target
X = train_processed.drop(['Survived', 'PassengerId'], axis=1)
y = train_processed['Survived']
X_test = test_processed.drop('PassengerId', axis=1)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# Fit the model
rf.fit(X_train, y_train)
# Make predictions
y_pred = rf.predict(X_val)
# Evaluate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.4f}')
# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Cross-validation mean score: {cv_scores.mean():.4f}')