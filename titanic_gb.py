import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import optuna
from statsmodels.stats.weightstats import DescrStatsW

np.random.seed(42)
# Set pandas display options
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
# Load data
train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')
print(train_df.head(100))
# Count the number of missing values in each column
print(train_df.isnull().sum())

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
    df_processed['Fare'] = df_processed.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
    df_processed['Embarked'] = df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0])
    df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_processed['Title'] = df_processed['Title'].replace(['Mlle', 'Ms'], 'Miss').replace(['Mme'], 'Mrs')

    def weighted_median_age(group):
        if group.notna().sum() == 0:  # If all NaN
            return 30
        stats = DescrStatsW(group.dropna(), weights=np.ones(len(group.dropna())))
        return stats.quantile(0.5, return_pandas=False)[0]

    # Fill missing Age with weighted median by Pclass and Title
    age_fill = df_processed.groupby(['Pclass', 'Title'])['Age'].apply(weighted_median_age)
    df_processed['Age'] = df_processed.apply(
        lambda row: age_fill.get((row['Pclass'], row['Title']), row['Age']) if pd.isna(row['Age']) else row['Age'],
        axis=1
    )

    df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')
    df_processed['Title'] = df_processed['Title'].replace(['Mlle', 'Ms'], 'Miss').replace(['Mme'], 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 2, "Master": 3, "Rare": 4}
    df_processed['Title'] = df_processed['Title'].map(title_mapping).fillna(0)
    df_processed['AgeCat'] = pd.cut(df_processed['Age'], bins=[0, 5, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4, 5])
    df_processed['AgeCat'] = df_processed['AgeCat'].astype(int)
    def fare_category(fr):
        if fr <= 7.91: return 1
        elif fr <= 14.454: return 2
        elif fr <= 31: return 3
        return 4
    df_processed['FareCat'] = df_processed['Fare'].apply(fare_category)
    df_processed['FareAge'] = df_processed['Fare'] * df_processed['Age']
    df_processed['ClassAge'] = df_processed['Pclass'] * df_processed['Age']
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

    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    df_processed['FamilySizeCat'] = df_processed['FamilySize'].map(lambda x: 1 if x == 1 else (2 if 5 > x >= 2 else (3 if 8 > x >= 5 else 4)))
    df_processed['Sex_Pclass_Embarked'] = df_processed['Sex'].astype(str) + '_' + df_processed['Pclass'].astype(str) + '_' + df_processed['Embarked'].astype(str)
    df_processed.drop(['Cabin', 'Ticket', 'Name', 'FamilySize', 'Fare', 'Age'], axis=1, inplace=True)
    le = LabelEncoder()
    for col in ['Sex']:
        df_processed[col] = le.fit_transform(df_processed[col])
    df_processed = pd.get_dummies(df_processed, columns=['AgeCat', 'Embarked', 'Title','FamilySizeCat', 'TicketPrefix', 'Sex_Pclass_Embarked'], drop_first=True)
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

# Optuna objective function for Gradient Boosting
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 20),
        'random_state': 42
    }
    model = GradientBoostingClassifier(**params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return cv_scores.mean()

# Optimize with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # 50 trials for thorough search

# Best parameters and model
best_params = study.best_params
print(f"Best Parameters: {best_params}")
best_model = GradientBoostingClassifier(**best_params, random_state=42)

# Evaluate on validation and full CV
best_model.fit(X_train, y_train)
val_score = accuracy_score(y_val, best_model.predict(X_val))
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"Tuned Gradient Boosting: Validation = {val_score:.4f}, CV Mean = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train on full data
best_model.fit(X, y)

# Submission
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': best_model.predict(X_test)
})
submission.to_csv('submission_gb_optuna.csv', index=False)
print("Submission file created: submission_gb_optuna.csv")