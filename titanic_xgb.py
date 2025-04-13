import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

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



### XGB Classifier
from xgboost import XGBClassifier
import xgboost
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', callbacks=[xgboost.callback.EarlyStopping(rounds=10, metric_name='logloss')], random_state=42, max_depth=5, n_estimators=200, learning_rate=0.05)
xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
    )
y_pred_xgb = xgb_model.predict(X_val_scaled)
###

accuracy_xgb = accuracy_score(y_val, y_pred_xgb)
print(f'Validation Accuracy of XGB: {accuracy_xgb:.4f}')
# Predict on test data
test_predictions_xgb = xgb_model.predict(X_test)
# Prepare submission file
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions_xgb
})
submission.to_csv('submission_xgb.csv', index=False)

# Plot feature importance for XGB
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Random Forest Feature Importance')
# sns.barplot(x='Importance', y='Feature', data=feature_importance)

# plt.subplot(1, 2, 2)
# plt.title('XGB Feature Importance')
# sns.barplot(x='Importance', y='Feature', data=xgb_importance)

# plt.tight_layout()
# plt.show()