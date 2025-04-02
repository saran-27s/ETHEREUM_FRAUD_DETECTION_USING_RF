import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('csv_files/transaction_dataset.csv', index_col=0)
print(f"Dataset shape: {df.shape}")

# Handle categorical columns
categories = df.select_dtypes('O').columns
print(f"Dropping categorical columns: {list(categories)}")
df.drop(df[categories], axis=1, inplace=True)

# Fill missing values with median
df.fillna(df.median(), inplace=True)

# Remove features with zero variance
no_var = df.var() == 0
if any(no_var):
    print(f"Dropping zero variance features: {list(df.var()[no_var].index)}")
    df.drop(df.var()[no_var].index, axis=1, inplace=True)

# Drop redundant or less important features
drop = ['total transactions (including tnx to create contract', 'total ether sent contracts', 
        'max val sent to contract', ' ERC20 avg val rec', ' ERC20 max val rec', ' ERC20 min val rec', 
        ' ERC20 uniq rec contract addr', 'max val sent', ' ERC20 avg val sent', ' ERC20 min val sent', 
        ' ERC20 max val sent', ' Total ERC20 tnxs', 'avg value sent to contract', 'Unique Sent To Addresses',
        'Unique Received From Addresses', 'total ether received', ' ERC20 uniq sent token name', 
        'min value received', 'min val sent', ' ERC20 uniq rec addr', 'min value sent to contract', 
        ' ERC20 uniq sent addr.1']

# Only drop columns that exist in the dataframe
drop_existing = [col for col in drop if col in df.columns]
if drop_existing:
    df.drop(drop_existing, axis=1, inplace=True)
    print(f"Dropped {len(drop_existing)} redundant features")

# Display updated dataset shape
print(f"Dataset shape after cleaning: {df.shape}")

# Split features and target
y = df['FLAG']
X = df.drop('FLAG', axis=1)
print(f"Features: {X.shape}, Target: {y.shape}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Normalize the features
norm = PowerTransformer()
norm_train_f = norm.fit_transform(X_train)
norm_test_f = norm.transform(X_test)

# Before SMOTE - Class distribution
print(f"Before SMOTE - Class distribution: {np.bincount(y_train)}")
fraud_nonfraud_before = np.bincount(y_train)
plt.figure(figsize=(8, 6))
plt.pie(fraud_nonfraud_before, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
plt.title('Fraud vs Non-Fraud Distribution (Before SMOTE)')
plt.savefig('images/fraud_distribution_before_smote.png')
plt.show()

# Apply SMOTE to handle class imbalance
oversample = SMOTE()
x_tr_resample, y_tr_resample = oversample.fit_resample(norm_train_f, y_train)
print(f"After SMOTE - Class distribution: {np.bincount(y_tr_resample)}")

# After SMOTE - Class distribution
fraud_nonfraud_after = np.bincount(y_tr_resample)
plt.figure(figsize=(8, 6))
plt.pie(fraud_nonfraud_after, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
plt.title('Fraud vs Non-Fraud Distribution (After SMOTE)')
plt.savefig('images/fraud_distribution_after_smote.png')
plt.show()

# Train Random Forest model
print("Training Random Forest model...")
RF = RandomForestClassifier(random_state=42, n_estimators=100)
RF.fit(x_tr_resample, y_tr_resample)

# Make predictions
preds_RF = RF.predict(norm_test_f)

# Evaluate model
print("\nModel Evaluation:")
print(classification_report(y_test, preds_RF))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, preds_RF)}")
print(f"ROC AUC Score: {roc_auc_score(y_test, preds_RF)}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': RF.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
plt.savefig('images/feature_importance.png')

# Save the model
model_filename = 'models/ethereum_fraud_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(RF, file)
print(f"\nModel saved as {model_filename}")

# Save the scaler for future predictions
scaler_filename = 'models/ethereum_fraud_scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(norm, file)
print(f"Scaler saved as {scaler_filename}")