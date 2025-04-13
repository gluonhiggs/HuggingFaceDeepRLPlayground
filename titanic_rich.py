import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter

np.random.seed(42)

# Load data
train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')

# Outlier removal
def detect_outlier(df, n, cols):
    outlier_indices = []
    for i in cols:
        Q1, Q3 = np.percentile(df[i].dropna(), [25, 75])
        IQR = Q3 - Q1
        step = 1.5 * IQR
        outlier_index_list = df[(df[i] < Q1 - step) | (df[i] > Q3 + step)].index
        outlier_indices.extend(outlier_index_list)
    outlier_indices = Counter(outlier_indices)
    return [k for k, v in outlier_indices.items() if v > n]

outliers = detect_outlier(train_df, 3, ['Age', 'SibSp', 'Parch', 'Fare'])
train_df = train_df.drop(outliers, axis=0).reset_index(drop=True)

# Combined preprocessing
total = pd.concat([train_df.drop('Survived', axis=1), test_df])

# Preprocessing function
def preprocess_data(df):
    df_processed = df.copy()
    
    # Fill missing values
    df_processed['Age'] = df_processed.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
    df_processed['Fare'] = df_processed.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
    df_processed['Embarked'] = df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0])
    
    # Titles (from Kaggle script)
    df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')
    df_processed['Title'] = df_processed['Title'].replace(['Mlle', 'Ms'], 'Miss').replace(['Mme'], 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 2, "Master": 3, "Rare": 4}
    df_processed['Title'] = df_processed['Title'].map(title_mapping).fillna(0)
    
    # Age Category (6 quantiles)
    df_processed['AgeCat'] = pd.cut(df_processed['Age'], bins=[0, 5, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4, 5])
    df_processed['AgeCat'] = df_processed['AgeCat'].astype(int)
    # Fare Category (custom bins)
    def fare_category(fr):
        if fr <= 7.91: return 1
        elif fr <= 14.454: return 2
        elif fr <= 31: return 3
        return 4
    df_processed['FareCat'] = df_processed['Fare'].apply(fare_category)
    
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
    
    # Family Size and Category
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    df_processed['FamilySizeCat'] = df_processed['FamilySize'].map(lambda x: 1 if x == 1 else (2 if 5 > x >= 2 else (3 if 8 > x >= 5 else 4)))
    
    # Interaction Feature (Fare_1_S equivalent)
    df_processed['Sex_Pclass_Embarked'] = df_processed['Sex'].astype(str) + '_' + df_processed['Pclass'].astype(str) + '_' + df_processed['Embarked'].astype(str)
    
    # Drop unneeded columns
    df_processed.drop(['Cabin', 'Ticket', 'Name', 'Age', 'FamilySize'], axis=1, inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in ['Sex', 'Embarked', 'Title', 'TicketPrefix', 'Sex_Pclass_Embarked']:
        df_processed[col] = le.fit_transform(df_processed[col])
    
    # Dummies for granular features
    df_processed = pd.get_dummies(df_processed, columns=['AgeCat', 'FareCat', 'FamilySizeCat', 'Title'])
    
    return df_processed

# Apply preprocessing
total_processed = preprocess_data(total)
train_processed = total_processed[:len(train_df)]
test_processed = total_processed[len(train_df):]
train_processed = train_processed.loc[train_df.index]  # Align with outlier-removed train

# Define features and target
X = train_processed.drop('PassengerId', axis=1)
y = train_df['Survived']
X_test = test_processed.drop('PassengerId', axis=1)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Model selection
models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

def fit_and_score(models, X_train, X_val, y_train, y_val, X, y):
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        # Validation accuracy
        val_score = accuracy_score(y_val, model.predict(X_val))
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        model_scores[name] = {'Validation': val_score, 'CV Mean': cv_mean, 'CV Std': cv_std}
        print(f"{name}: Validation = {val_score:.4f}, CV Mean = {cv_mean:.4f} ± {cv_std:.4f}")
    return model_scores

# Evaluate models
model_scores = fit_and_score(models, X_train, X_val, y_train, y_val, X, y)

# Choose the best model based on CV Mean
best_model_name = max(model_scores, key=lambda k: model_scores[k]['CV Mean'])
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with CV Mean Score: {model_scores[best_model_name]['CV Mean']:.4f}")

# Train best model on full data
best_model.fit(X, y)

# Submission
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': best_model.predict(X_test)
})
submission.to_csv('submission_best_model.csv', index=False)
print("Submission file created: submission_best_model.csv")