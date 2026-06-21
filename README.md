# Detailed Summary of `model.py`

This notebook provides a thorough explanation of the `model.py` script, focusing on the libraries used, their purposes, and their specific applications within the code. We'll explore the key roles of **Pandas**, **NumPy**, **Scikit-learn**, and other libraries in this machine learning pipeline for fraud detection.

## Overview of the Machine Learning Pipeline

`model.py` implements a complete machine learning workflow for fraud detection in Ethereum transactions, including:

1. Data loading and preprocessing
2. Feature engineering and selection
3. Handling class imbalance through SMOTE
4. Model training (Random Forest)
5. Performance evaluation
6. Model persistence

## Libraries Used

### 1. **Pandas**
**Purpose**: Data manipulation and analysis framework providing powerful data structures like `DataFrame` and `Series`.

**Where It Is Used**:
- **Data Loading**:
  ```python
  df = pd.read_csv('csv_files/transaction_dataset.csv', index_col=0)
  ```
- **Data Cleaning**:
  ```python
  # Removing categorical features
  categories = df.select_dtypes('O').columns
  df.drop(df[categories], axis=1, inplace=True)
  
  # Handling missing values
  df.fillna(df.median(), inplace=True)
  
  # Removing zero-variance features
  no_var = df.var() == 0
  df.drop(df.var()[no_var].index, axis=1, inplace=True)
  ```
- **Feature Engineering**:
  ```python
  # Removing less important or redundant features
  drop_existing = [col for col in drop if col in df.columns]
  df.drop(drop_existing, axis=1, inplace=True)
  ```
- **Dataset Preparation**:
  ```python
  # Splitting features and target variable
  y = df['FLAG']  # Target (fraud indicator)
  X = df.drop('FLAG', axis=1)  # Features
  ```
- **Results Analysis**:
  ```python
  # Creating a DataFrame to display feature importance
  feature_importance = pd.DataFrame({
      'Feature': X.columns,
      'Importance': RF.feature_importances_
  }).sort_values(by='Importance', ascending=False)
  ```

**Why It Is Important**:
- Provides intuitive interfaces for complex data operations
- Enables efficient data transformation and cleaning
- Simplifies feature selection and engineering
- Offers seamless integration with visualization libraries

### 2. **NumPy**
**Purpose**: Foundation for numerical computing in Python, providing efficient array operations and mathematical functions.

**Where It Is Used**:
- **Array Operations**:
  ```python
  # Used internally by scikit-learn when transforming data
  norm_train_f = norm.fit_transform(X_train)
  norm_test_f = norm.transform(X_test)
  ```
- **Statistical Analysis**:
  ```python
  # Class distribution analysis
  np.bincount(y_train)  # Before resampling
  np.bincount(y_tr_resample)  # After resampling
  
  # Used internally for median calculations
  df.fillna(df.median(), inplace=True)
  ```

**Why It Is Important**:
- Provides the foundation for scientific computing in Python
- Enables efficient numerical operations on large datasets
- Underpins most machine learning libraries, including Scikit-learn
- Optimized for performance with vectorized operations

### 3. **Scikit-learn**
**Purpose**: Comprehensive machine learning library offering tools for data preprocessing, model training, and evaluation.

**Where It Is Used**:
- **Data Splitting**:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.4, random_state=123
  )
  ```
- **Feature Normalization**:
  ```python
  # Using PowerTransformer for feature normalization
  norm = PowerTransformer()
  norm_train_f = norm.fit_transform(X_train)
  norm_test_f = norm.transform(X_test)
  ```
- **Model Training**:
  ```python
  # Training Random Forest classifier
  RF = RandomForestClassifier(random_state=42, n_estimators=100)
  RF.fit(x_tr_resample, y_tr_resample)
  ```
- **Prediction**:
  ```python
  # Making predictions on test data
  preds_RF = RF.predict(norm_test_f)
  ```
- **Model Evaluation**:
  ```python
  # Evaluating model performance
  print(classification_report(y_test, preds_RF))
  print(confusion_matrix(y_test, preds_RF))
  print(roc_auc_score(y_test, preds_RF))
  ```
- **Feature Importance Analysis**:
  ```python
  # Extracting feature importance from the model
  feature_importance = pd.DataFrame({
      'Feature': X.columns,
      'Importance': RF.feature_importances_
  }).sort_values(by='Importance', ascending=False)
  ```

**Why It Is Important**:
- Provides standardized implementations of machine learning algorithms
- Offers consistent APIs for model training and evaluation
- Includes robust preprocessing tools
- Facilitates model selection and hyperparameter tuning

### 4. **Matplotlib**
**Purpose**: Primary visualization library for creating static, interactive, and animated plots.

**Where It Is Used**:
- **Class Distribution Visualization**:
  ```python
  # Visualizing class distribution before SMOTE
  plt.figure(figsize=(8, 6))
  plt.pie(fraud_nonfraud_before, labels=['Non-Fraud', 'Fraud'], 
          autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
  plt.title('Fraud Distribution Before SMOTE')
  plt.savefig('images/fraud_distribution_before_smote.png')
  
  # Visualizing class distribution after SMOTE
  plt.figure(figsize=(8, 6))
  plt.pie(fraud_nonfraud_after, labels=['Non-Fraud', 'Fraud'], 
          autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
  plt.title('Fraud Distribution After SMOTE')
  plt.savefig('images/fraud_distribution_after_smote.png')
  ```
- **Formatting and Saving Plots**:
  ```python
  plt.tight_layout()
  plt.savefig('images/feature_importance.png')
  ```

**Why It Is Important**:
- Enables data visualization for exploratory data analysis (EDA)
- Helps communicate model insights through informative graphics
- Allows customization of plots for different purposes
- Integrates with other libraries (Seaborn, Pandas)

### 5. **Seaborn**
**Purpose**: High-level interface for creating attractive statistical graphics based on Matplotlib.

**Where It Is Used**:
- **Feature Importance Visualization**:
  ```python
  plt.figure(figsize=(10, 8))
  sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
  plt.title('Top 10 Important Features')
  plt.tight_layout()
  plt.savefig('images/feature_importance.png')
  ```

**Why It Is Important**:
- Creates visually appealing statistical visualizations
- Simplifies creation of complex plots like bar charts, heatmaps, and distribution plots
- Integrates well with Pandas DataFrames
- Enhances data exploration and results presentation

### 6. **Pickle**
**Purpose**: Python module for object serialization and deserialization.

**Where It Is Used**:
- **Model Persistence**:
  ```python
  # Saving the trained Random Forest model
  with open('models/ethereum_fraud_model.pkl', 'wb') as file:
      pickle.dump(RF, file)
  
  # Saving the data transformation pipeline
  with open('models/ethereum_fraud_scaler.pkl', 'wb') as file:
      pickle.dump(norm, file)
  ```

**Why It Is Important**:
- Enables saving and loading of trained models
- Preserves preprocessing transformations for consistent inference
- Allows for model deployment in production environments
- Facilitates sharing models between different systems

### 7. **Imbalanced-learn (SMOTE)**
**Purpose**: Library for handling imbalanced datasets through resampling techniques.

**Where It Is Used**:
- **Class Imbalance Handling**:
  ```python
  # Applying SMOTE to address class imbalance
  oversample = SMOTE()
  x_tr_resample, y_tr_resample = oversample.fit_resample(norm_train_f, y_train)
  
  # Checking class distribution after resampling
  print("Class distribution after SMOTE:", np.bincount(y_tr_resample))
  ```

**Why It Is Important**:
- Addresses the common problem of class imbalance in fraud detection
- Improves model performance on minority classes
- Helps prevent bias toward majority class predictions
- Integrates seamlessly with scikit-learn workflows

## Complete Machine Learning Pipeline

The `model.py` script implements a comprehensive machine learning pipeline for fraud detection:

1. **Data Loading and Cleaning**:
   - Load data from CSV
   - Remove categorical features
   - Handle missing values
   - Remove zero-variance features

2. **Feature Engineering**:
   - Remove redundant or less important features
   - Split features and target variable

3. **Data Preparation**:
   - Split data into training and testing sets
   - Normalize features using PowerTransformer

4. **Class Imbalance Handling**:
   - Apply SMOTE to balance the training dataset
   - Verify class distribution before and after resampling

5. **Model Training**:
   - Train a Random Forest classifier on the resampled data

6. **Model Evaluation**:
   - Generate predictions on the test set
   - Calculate classification metrics (precision, recall, F1-score)
   - Generate confusion matrix
   - Calculate ROC AUC score

7. **Feature Importance Analysis**:
   - Extract and visualize feature importance

8. **Model Persistence**:
   - Save the trained model and preprocessing pipeline for future use

## Key Libraries Comparison

| **Library** | **Primary Role** | **Key Functions Used** | **Importance in Pipeline** |
|-------------|------------------|------------------------|----------------------------|
| **Pandas**  | Data manipulation | `read_csv`, `drop`, `fillna`, `DataFrame` | High - Data loading, cleaning, and feature selection |
| **NumPy**   | Numerical computing | Array operations, statistical functions | Medium - Backend for scikit-learn and statistical operations |
| **Scikit-learn** | Machine learning | `train_test_split`, `PowerTransformer`, `RandomForestClassifier`, evaluation metrics | High - Core ML functionality including preprocessing, training, and evaluation |
| **Matplotlib** | Visualization | `pie`, `figure`, `savefig` | Medium - Visualizing results and data distributions |
| **Seaborn** | Statistical visualization | `barplot` | Medium - Enhanced visualization of feature importance |
| **Pickle** | Object serialization | `dump` | Medium - Model persistence for deployment |
| **Imbalanced-learn** | Class imbalance handling | `SMOTE` | High - Critical for addressing class imbalance in fraud detection |

## Conclusion

The `model.py` script demonstrates a well-structured machine learning pipeline for fraud detection in Ethereum transactions. It leverages popular data science libraries to handle every stage of the machine learning workflow, from data preparation to model evaluation and persistence. The combination of these libraries enables efficient processing of structured data, handling of class imbalance, accurate model training, and comprehensive evaluation of results.
