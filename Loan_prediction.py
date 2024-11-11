import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../Loan_Status_Prediction/loan_prediction.csv')

# Display the top 5 rows of the dataset
print(data.head())

# Check the last 5 rows of the dataset
print(data.tail())

# Find the shape of the dataset (number of rows and columns)
print("Number of Rows:", data.shape[0])
print("Number of Columns:", data.shape[1])

# Get information about the dataset
data.info()

# Check for null values in the dataset
print(data.isnull().sum())
print(data.isnull().sum() * 100 / len(data))

# Handling missing values
data = data.drop('Loan_ID', axis=1)
columns = ['Gender', 'Dependents', 'LoanAmount', 'Loan_Amount_Term']
data = data.dropna(subset=columns)

# Fill missing values
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

# Replace values in 'Dependents'
data['Dependents'] = data['Dependents'].replace(to_replace="3+", value='4')

# Encoding categorical columns
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0}).astype('int')
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0}).astype('int')
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural': 0, 'Semiurban': 2, 'Urban': 1}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0}).astype('int')

# Store feature matrix in X and response (target) in vector y
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Handling class imbalance using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Feature scaling
cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = StandardScaler()
X_resampled[cols] = scaler.fit_transform(X_resampled[cols])

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=42)

# Model validation function
model_df = {}


def model_val(model, X, y):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"{model} accuracy is {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    score = cross_val_score(model, X, y, cv=5)
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model] = round(np.mean(score) * 100, 2)


# Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model_val(model, X_resampled, y_resampled)

# Support Vector Classifier
from sklearn import svm

model = svm.SVC()
model_val(model, X_resampled, y_resampled)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model_val(model, X_resampled, y_resampled)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model_val(model, X_resampled, y_resampled)

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model_val(model, X_resampled, y_resampled)

# Hyperparameter tuning for Logistic Regression
log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ['liblinear']}
rs_log_reg = RandomizedSearchCV(LogisticRegression(), param_distributions=log_reg_grid, n_iter=20, cv=5, verbose=True)
rs_log_reg.fit(X_resampled, y_resampled)
print(rs_log_reg.best_score_)
print(rs_log_reg.best_params_)

# Hyperparameter tuning for SVC
svc_grid = {'C': [0.25, 0.50, 0.75, 1], "kernel": ["linear"]}
rs_svc = RandomizedSearchCV(svm.SVC(), param_distributions=svc_grid, cv=5, n_iter=20, verbose=True)
rs_svc.fit(X_resampled, y_resampled)
print(rs_svc.best_score_)
print(rs_svc.best_params_)

# Hyperparameter tuning for Random Forest Classifier
rf_grid = {
    'n_estimators': np.arange(10, 1000, 10),
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 3, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 20, 50, 100],
    'min_samples_leaf': [1, 2, 5, 10]
}

rs_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid, cv=5, n_iter=20, verbose=True)
rs_rf.fit(X_resampled, y_resampled)
print(rs_rf.best_score_)
print(rs_rf.best_params_)

# Plotting model comparison
sns.barplot(x=list(model_df.keys()), y=list(model_df.values()))
plt.title('Model Comparison')
plt.ylabel('Average Cross Validation Score (%)')
plt.xticks(rotation=45)
plt.show()

# Saving the model
final_rf = RandomForestClassifier(
    n_estimators=rs_rf.best_params_['n_estimators'],
    min_samples_split=rs_rf.best_params_['min_samples_split'],
    min_samples_leaf=rs_rf.best_params_['min_samples_leaf'],
    max_features=rs_rf.best_params_['max_features'],
    max_depth=rs_rf.best_params_['max_depth']
)

final_rf.fit(X_resampled, y_resampled)

# Save the model and scaler
joblib.dump(final_rf, 'loan_status_predict.pkl')
joblib.dump(scaler, 'scaler.pkl')

#Plotting ROC curve
y_pred_prob = final_rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

plt.show()
