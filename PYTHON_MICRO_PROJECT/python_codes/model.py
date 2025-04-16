import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, roc_curve, auc
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

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

# ----- 1. FEATURE CORRELATION ANALYSIS -----
# Create a correlation matrix and visualization
plt.figure(figsize=(16, 12))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png', dpi=300)
plt.close()


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

# ----- 4. CLASS IMBALANCE VISUALIZATION -----
# Before SMOTE - Class distribution
fraud_nonfraud_before = np.bincount(y_train)
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.pie(fraud_nonfraud_before, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, 
        colors=['skyblue', 'orange'], explode=[0, 0.1])
plt.title('Before SMOTE')

# Apply SMOTE to handle class imbalance
oversample = SMOTE()
x_tr_resample, y_tr_resample = oversample.fit_resample(norm_train_f, y_train)

# After SMOTE - Class distribution
fraud_nonfraud_after = np.bincount(y_tr_resample)
plt.subplot(1, 2, 2)
plt.pie(fraud_nonfraud_after, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, 
        colors=['skyblue', 'orange'], explode=[0, 0.1])
plt.title('After SMOTE')

plt.suptitle('Class Distribution: Fraud vs Non-Fraud', fontsize=16)
plt.tight_layout()
plt.savefig('images/class_imbalance_comparison.png', dpi=300)
plt.close()

# ----- 5. TRANSACTION VOLUME ANALYSIS -----
# Create a bar chart comparing transaction volume between fraud and non-fraud
plt.figure(figsize=(12, 8))
transaction_features = ['Sent tnx', 'Received Tnx', 'Number of Created Contracts']

# Calculate means grouped by FLAG
transaction_means = df.groupby('FLAG')[transaction_features].mean().reset_index()

# Reshape for seaborn
transaction_melted = pd.melt(transaction_means, id_vars='FLAG', 
                            value_vars=transaction_features,
                            var_name='Transaction Type', value_name='Average Count')

# Plot
sns.barplot(x='Transaction Type', y='Average Count', hue='FLAG', data=transaction_melted, 
           palette=['skyblue', 'orange'])
plt.title('Average Transaction Volume: Fraud vs Non-Fraud', fontsize=16)
plt.xlabel('Transaction Type', fontsize=12)
plt.ylabel('Average Count', fontsize=12)
plt.legend(title='Account Type', labels=['Non-Fraud', 'Fraud'])
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('images/transaction_volume_comparison.png', dpi=300)
plt.close()

# Train Random Forest model
print("Training Random Forest model...")
RF = RandomForestClassifier(random_state=42, n_estimators=100)
RF.fit(x_tr_resample, y_tr_resample)

# Make predictions
preds_RF = RF.predict(norm_test_f)
probs_RF = RF.predict_proba(norm_test_f)[:, 1]

from sklearn.metrics import accuracy_score, recall_score

# Calculate metrics
accuracy = accuracy_score(y_test, preds_RF)
recall = recall_score(y_test, preds_RF)
roc_auc = roc_auc_score(y_test, probs_RF)

# Print metrics to console
print(f"Accuracy Score: {accuracy:.4f}")
print(f"Recall Score: {recall:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# ----- 6. ROC CURVE VISUALIZATION -----
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, probs_RF)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('images/roc_curve.png', dpi=300)
plt.close()

# ----- 7. CONFUSION MATRIX VISUALIZATION -----
cm = confusion_matrix(y_test, preds_RF)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks([0.5, 1.5], ['Non-Fraud', 'Fraud'])
plt.yticks([0.5, 1.5], ['Non-Fraud', 'Fraud'])
plt.tight_layout()
plt.savefig('images/confusion_matrix_visualization.png', dpi=300)
plt.close()

# ----- 8. FEATURE IMPORTANCE VISUALIZATION -----
# Create a more detailed feature importance visualization
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': RF.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot top 15 features
top_n = 15
plt.figure(figsize=(12, 10))
ax = sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n), palette='viridis')
plt.title('Top 15 Most Important Features for Fraud Detection', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)

# Add importance values to the end of each bar
for i, v in enumerate(feature_importance['Importance'].head(top_n)):
    ax.text(v + 0.001, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.savefig('images/detailed_feature_importance.png', dpi=300)
plt.close()

# ----- 9. MODEL PERFORMANCE METRICS VISUALIZATION -----
# Get classification report as a dictionary
report = classification_report(y_test, preds_RF, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Plot precision, recall, and f1-score
plt.figure(figsize=(12, 8))
metrics_df = report_df.iloc[0:2][['precision', 'recall', 'f1-score']]
metrics_df.index = ['Non-Fraud', 'Fraud']
metrics_df.plot(kind='bar', colormap='viridis')
plt.title('Model Performance Metrics by Class', fontsize=16)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim([0, 1])
plt.xticks(rotation=0)
plt.legend(title='Metric')
plt.tight_layout()
plt.savefig('images/performance_metrics.png', dpi=300)
plt.close()

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

print("\nAll visualizations have been saved in the 'images' directory.")