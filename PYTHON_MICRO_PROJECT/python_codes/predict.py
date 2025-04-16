import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_collector import get_data

def predict_fraud(wallet_data_path='csv_files/wallet_data.csv'):
    """
    Make fraud predictions on wallet data using the pre-trained model
    
    Parameters:
    wallet_data_path (str): Path to CSV file containing wallet data
    
    Returns:
    pd.DataFrame: DataFrame with wallet addresses and fraud predictions
    """
    # Load the pre-trained model and scaler
    try:
        print("Loading pre-trained model and scaler...")
        with open('models/ethereum_fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/ethereum_fraud_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure that the model and scaler files exist in the 'models' directory.")
        return
    
    # Load wallet data
    print(f"Loading wallet data from {wallet_data_path}...")
    wallet_data = pd.read_csv(wallet_data_path, index_col=0)
    print(f"Wallet data shape: {wallet_data.shape}")
    
    
    # Store wallet addresses if present
    addresses = None
    if 'Address' in wallet_data.columns:
        addresses = wallet_data['Address'].copy()
        wallet_data = wallet_data.drop('Address', axis=1)
    
    # Handle missing values
    wallet_data.fillna(wallet_data.median(), inplace=True)
    
    # Drop FLAG column if present (for testing purposes)
    true_labels = None
    if 'FLAG' in wallet_data.columns:
        true_labels = wallet_data['FLAG'].copy()
        wallet_data = wallet_data.drop('FLAG', axis=1)
    
    # Align feature names with training data
    required_features = scaler.feature_names_in_  # Features used during training
    for feature in required_features:
        if feature not in wallet_data.columns:
            wallet_data[feature] = 0  # Add missing features with default value 0
    wallet_data = wallet_data[required_features]  # Select only required features
    
    # Scale features
    wallet_data_scaled = scaler.transform(wallet_data)
    
    # Make predictions
    print("Making predictions...")
    fraud_probs = model.predict_proba(wallet_data_scaled)[:, 1]
    fraud_preds = model.predict(wallet_data_scaled)
    
    # Create results dataframe
    if addresses is not None:
        results = pd.DataFrame({
            'Address': addresses,
            'Fraud_Prediction': fraud_preds,
            'Fraud_Probability': fraud_probs
        })
    else:
        results = pd.DataFrame({
            'Fraud_Prediction': fraud_preds,
            'Fraud_Probability': fraud_probs
        })
    
    # Sort by fraud probability
    results = results.sort_values('Fraud_Probability', ascending=False).reset_index(drop=True)
    
    # Save results
    output_file = 'csv_files/fraud_predictions.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results.to_csv(output_file, index=True)
    print("Predictions saved to 'csv_files/fraud_predictions.csv'")
    
    # Display top potential fraudsters
    print(results.head(10))
    
    # Plot fraud probability distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(fraud_probs, bins=50)
    plt.title('Distribution of Fraud Probabilities')
    plt.xlabel('Probability of Fraud')
    plt.ylabel('Count')
    plt.savefig('images/fraud_probability_distribution.png')
    
    # Evaluate against true labels if available
    if true_labels is not None:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nModel performance on this dataset:")
        print(classification_report(true_labels, fraud_preds))
        print(f"Confusion Matrix:\n{confusion_matrix(true_labels, fraud_preds)}")
    
    return results

if __name__ == "__main__":
    wallet_address=input("Enter the wallet address to predict : ")
    get_data(wallet_address)
    predict_fraud()