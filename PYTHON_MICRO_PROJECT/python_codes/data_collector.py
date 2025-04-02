import requests
import pandas as pd
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os

# Etherscan API key
ETHERSCAN_API_KEY = "848HZHG8QKCE4DIMBV7P13CDSUARHXDX4T"
BASE_URL = "https://api.etherscan.io/api"

def get_normal_transactions(address):
    """Fetch normal transactions for an address"""
    params = {
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'sort': 'asc',
        'apikey': ETHERSCAN_API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == '1':
            return pd.DataFrame(data['result'])
        else:
            print(f"Error fetching normal transactions: {data['message']}")
            return pd.DataFrame()
    else:
        print(f"API request failed with status code: {response.status_code}")
        return pd.DataFrame()

def get_internal_transactions(address):
    """Fetch internal transactions for an address"""
    params = {
        'module': 'account',
        'action': 'txlistinternal',
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'sort': 'asc',
        'apikey': ETHERSCAN_API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == '1':
            return pd.DataFrame(data['result'])
        else:
            print(f"Error fetching internal transactions: {data['message']}")
            return pd.DataFrame()
    else:
        print(f"API request failed with status code: {response.status_code}")
        return pd.DataFrame()

def get_erc20_transactions(address):
    """Fetch ERC20 token transactions for an address"""
    params = {
        'module': 'account',
        'action': 'tokentx',
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'sort': 'asc',
        'apikey': ETHERSCAN_API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == '1':
            return pd.DataFrame(data['result'])
        else:
            print(f"Error fetching ERC20 transactions: {data['message']}")
            return pd.DataFrame()
    else:
        print(f"API request failed with status code: {response.status_code}")
        return pd.DataFrame()

def get_account_balance(address):
    """Get current ETH balance for an address"""
    params = {
        'module': 'account',
        'action': 'balance',
        'address': address,
        'tag': 'latest',
        'apikey': ETHERSCAN_API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == '1':
            return float(data['result']) / 10**18  # Convert wei to ETH
        else:
            print(f"Error fetching balance: {data['message']}")
            return 0
    else:
        print(f"API request failed with status code: {response.status_code}")
        return 0

def extract_features(address):
    """Extract features for a given Ethereum address"""
    print(f"Processing address: {address}")
    features = {'Address': address}
    
    # Respect API rate limits (5 calls/sec)
    time.sleep(0.2)
    
    # Get transactions
    normal_txs = get_normal_transactions(address)
    time.sleep(0.2)
    internal_txs = get_internal_transactions(address)
    time.sleep(0.2)
    erc20_txs = get_erc20_transactions(address)
    time.sleep(0.2)
    balance = get_account_balance(address)
    
    if normal_txs.empty and internal_txs.empty and erc20_txs.empty:
        print(f"No transactions found for address: {address}")
        return None
    
    # Process normal transactions
    if not normal_txs.empty:
        # Convert timestamps to datetime
        normal_txs['timeStamp'] = pd.to_datetime(normal_txs['timeStamp'].astype(int), unit='s')
        
        # Sent transactions (from address matches our address)
        sent_txs = normal_txs[normal_txs['from'].str.lower() == address.lower()]
        # Received transactions (to address matches our address)
        received_txs = normal_txs[normal_txs['to'].str.lower() == address.lower()]
        
        # Calculate time differences between transactions
        if len(sent_txs) > 1:
            sent_diffs = sent_txs['timeStamp'].diff().dropna()
            features['Avg min between sent tnx'] = sent_diffs.mean().total_seconds() / 60
        else:
            features['Avg min between sent tnx'] = 0
            
        if len(received_txs) > 1:
            received_diffs = received_txs['timeStamp'].diff().dropna()
            features['Avg min between received tnx'] = received_diffs.mean().total_seconds() / 60
        else:
            features['Avg min between received tnx'] = 0
        
        # Time difference between first and last transaction
        if not normal_txs.empty:
            first_tx = normal_txs['timeStamp'].min()
            last_tx = normal_txs['timeStamp'].max()
            features['Time Diff between first and last (Mins)'] = (last_tx - first_tx).total_seconds() / 60
        else:
            features['Time Diff between first and last (Mins)'] = 0
        
        # Count sent and received transactions
        features['Sent tnx'] = len(sent_txs)
        features['Received Tnx'] = len(received_txs)
        
        # Count created contracts
        features['Number of Created Contracts'] = len(normal_txs[normal_txs['to'] == ''])
        
        # Count unique addresses
        features['Unique Received From Addresses'] = received_txs['from'].nunique()
        features['Unique Sent To Addresses'] = sent_txs['to'].nunique()
        
        # Value statistics
        if not received_txs.empty:
            received_values = received_txs['value'].astype(float) / 10**18
            features['min value received'] = received_values.min() if len(received_values) > 0 else 0
            features['max value received'] = received_values.max() if len(received_values) > 0 else 0
            features['avg val received'] = received_values.mean() if len(received_values) > 0 else 0
            features['total ether received'] = received_values.sum() if len(received_values) > 0 else 0
        else:
            features['min value received'] = 0
            features['max value received'] = 0
            features['avg val received'] = 0
            features['total ether received'] = 0
        
        if not sent_txs.empty:
            sent_values = sent_txs['value'].astype(float) / 10**18
            features['min val sent'] = sent_values.min() if len(sent_values) > 0 else 0
            features['max val sent'] = sent_values.max() if len(sent_values) > 0 else 0
            features['avg val sent'] = sent_values.mean() if len(sent_values) > 0 else 0
            features['total Ether sent'] = sent_values.sum() if len(sent_values) > 0 else 0
        else:
            features['min val sent'] = 0
            features['max val sent'] = 0
            features['avg val sent'] = 0
            features['total Ether sent'] = 0
        
        # Contract interactions
        contract_txs = sent_txs[sent_txs['to'].str.lower() != '']  # Filter out contract creation txs
        if not contract_txs.empty:
            contract_values = contract_txs['value'].astype(float) / 10**18
            features['min value sent to contract'] = contract_values.min() if len(contract_values) > 0 else 0
            features['max val sent to contract'] = contract_values.max() if len(contract_values) > 0 else 0
            features['avg value sent to contract'] = contract_values.mean() if len(contract_values) > 0 else 0
            features['total ether sent contracts'] = contract_values.sum() if len(contract_values) > 0 else 0
        else:
            features['min value sent to contract'] = 0
            features['max val sent to contract'] = 0
            features['avg value sent to contract'] = 0
            features['total ether sent contracts'] = 0
    else:
        # Set defaults for normal transaction features
        for feature in ['Avg min between sent tnx', 'Avg min between received tnx', 
                       'Time Diff between first and last (Mins)', 'Sent tnx', 
                       'Received Tnx', 'Number of Created Contracts', 
                       'Unique Received From Addresses', 'Unique Sent To Addresses',
                       'min value received', 'max value received', 'avg val received',
                       'min val sent', 'max val sent', 'avg val sent',
                       'min value sent to contract', 'max val sent to contract', 
                       'avg value sent to contract', 'total Ether sent',
                       'total ether received', 'total ether sent contracts']:
            features[feature] = 0
    
    # Process ERC20 transactions
    if not erc20_txs.empty:
        erc20_txs['timeStamp'] = pd.to_datetime(erc20_txs['timeStamp'].astype(int), unit='s')
        erc20_sent = erc20_txs[erc20_txs['from'].str.lower() == address.lower()]
        erc20_received = erc20_txs[erc20_txs['to'].str.lower() == address.lower()]
        
        features[' Total ERC20 tnxs'] = len(erc20_txs)
        
        # ERC20 token values
        if not erc20_received.empty:
            erc20_rec_values = erc20_received['value'].astype(float) / 10**18  # Simplified, should consider token decimals
            features[' ERC20 total Ether received'] = erc20_rec_values.sum() if len(erc20_rec_values) > 0 else 0
            features[' ERC20 min val rec'] = erc20_rec_values.min() if len(erc20_rec_values) > 0 else 0
            features[' ERC20 max val rec'] = erc20_rec_values.max() if len(erc20_rec_values) > 0 else 0
            features[' ERC20 avg val rec'] = erc20_rec_values.mean() if len(erc20_rec_values) > 0 else 0
        else:
            features[' ERC20 total Ether received'] = 0
            features[' ERC20 min val rec'] = 0
            features[' ERC20 max val rec'] = 0
            features[' ERC20 avg val rec'] = 0
        
        if not erc20_sent.empty:
            erc20_sent_values = erc20_sent['value'].astype(float) / 10**18  # Simplified
            features[' ERC20 total ether sent'] = erc20_sent_values.sum() if len(erc20_sent_values) > 0 else 0
            features[' ERC20 min val sent'] = erc20_sent_values.min() if len(erc20_sent_values) > 0 else 0
            features[' ERC20 max val sent'] = erc20_sent_values.max() if len(erc20_sent_values) > 0 else 0
            features[' ERC20 avg val sent'] = erc20_sent_values.mean() if len(erc20_sent_values) > 0 else 0
        else:
            features[' ERC20 total ether sent'] = 0
            features[' ERC20 min val sent'] = 0
            features[' ERC20 max val sent'] = 0
            features[' ERC20 avg val sent'] = 0
        
        # Contract interactions with ERC20
        erc20_contract_txs = erc20_sent[erc20_sent['to'].str.lower() != address.lower()]
        if not erc20_contract_txs.empty:
            erc20_contract_values = erc20_contract_txs['value'].astype(float) / 10**18
            features[' ERC20 total Ether sent contract'] = erc20_contract_values.sum() if len(erc20_contract_values) > 0 else 0
            features[' ERC20 min val sent contract'] = erc20_contract_values.min() if len(erc20_contract_values) > 0 else 0
            features[' ERC20 max val sent contract'] = erc20_contract_values.max() if len(erc20_contract_values) > 0 else 0
            features[' ERC20 avg val sent contract'] = erc20_contract_values.mean() if len(erc20_contract_values) > 0 else 0
        else:
            features[' ERC20 total Ether sent contract'] = 0
            features[' ERC20 min val sent contract'] = 0
            features[' ERC20 max val sent contract'] = 0
            features[' ERC20 avg val sent contract'] = 0
        
        # Unique addresses
        features[' ERC20 uniq sent addr'] = erc20_sent['to'].nunique()
        features[' ERC20 uniq rec addr'] = erc20_received['from'].nunique()
        features[' ERC20 uniq sent addr.1'] = erc20_sent['to'].nunique()  # Duplicate for model compatibility
        features[' ERC20 uniq rec contract addr'] = erc20_contract_txs['to'].nunique() if not erc20_contract_txs.empty else 0
        
        # Time between transactions
        if len(erc20_sent) > 1:
            erc20_sent_diffs = erc20_sent['timeStamp'].diff().dropna()
            features[' ERC20 avg time between sent tnx'] = erc20_sent_diffs.mean().total_seconds() / 60
        else:
            features[' ERC20 avg time between sent tnx'] = 0
        
        if len(erc20_received) > 1:
            erc20_rec_diffs = erc20_received['timeStamp'].diff().dropna()
            features[' ERC20 avg time between rec tnx'] = erc20_rec_diffs.mean().total_seconds() / 60
        else:
            features[' ERC20 avg time between rec tnx'] = 0
        
        # Token names
        if not erc20_txs.empty and 'tokenName' in erc20_txs.columns:
            features[' ERC20 uniq sent token name'] = erc20_sent['tokenName'].nunique() if not erc20_sent.empty else 0
            features[' ERC20 uniq rec token name'] = erc20_received['tokenName'].nunique() if not erc20_received.empty else 0
            
            # Most used token types
            if not erc20_sent.empty:
                features[' ERC20_most_sent_token_type'] = erc20_sent['tokenName'].value_counts().index[0] if len(erc20_sent['tokenName'].value_counts()) > 0 else "None"
            else:
                features[' ERC20_most_sent_token_type'] = "None"
                
            if not erc20_received.empty:
                features[' ERC20_most_rec_token_type'] = erc20_received['tokenName'].value_counts().index[0] if len(erc20_received['tokenName'].value_counts()) > 0 else "None"
            else:
                features[' ERC20_most_rec_token_type'] = "None"
        else:
            features[' ERC20 uniq sent token name'] = 0
            features[' ERC20 uniq rec token name'] = 0
            features[' ERC20_most_sent_token_type'] = "None"
            features[' ERC20_most_rec_token_type'] = "None"
    else:
        # Set defaults for ERC20 features
        for feature in [' Total ERC20 tnxs', ' ERC20 total Ether received', ' ERC20 total ether sent',
                        ' ERC20 total Ether sent contract', ' ERC20 uniq sent addr', ' ERC20 uniq rec addr',
                        ' ERC20 uniq sent addr.1', ' ERC20 uniq rec contract addr', ' ERC20 avg time between sent tnx',
                        ' ERC20 avg time between rec tnx', ' ERC20 avg time between rec 2 tnx',
                        ' ERC20 avg time between contract tnx', ' ERC20 min val rec', ' ERC20 max val rec',
                        ' ERC20 avg val rec', ' ERC20 min val sent', ' ERC20 max val sent', ' ERC20 avg val sent',
                        ' ERC20 min val sent contract', ' ERC20 max val sent contract', ' ERC20 avg val sent contract',
                        ' ERC20 uniq sent token name', ' ERC20 uniq rec token name']:
            features[feature] = 0
        features[' ERC20_most_sent_token_type'] = "None"
        features[' ERC20_most_rec_token_type'] = "None"
    
    # Add total transactions
    features['total transactions (including tnx to create contract'] = features['Sent tnx'] + features['Received Tnx']
    
    # Add balance
    features['total ether balance'] = balance
    
    return features

def fetch_addresses_from_etherscan(n=100):
    """Fetch recent addresses from Etherscan for analysis"""
    # Get recent blocks
    params = {
        'module': 'proxy',
        'action': 'eth_blockNumber',
        'apikey': ETHERSCAN_API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        current_block = int(data['result'], 16)
        
        # Start from 10 blocks ago to get recent transactions
        start_block = current_block - 10
        
        # Get transactions from these blocks
        unique_addresses = set()
        
        for block_num in range(start_block, current_block + 1):
            # Respect API rate limits
            time.sleep(0.2)
            
            params = {
                'module': 'proxy',
                'action': 'eth_getBlockByNumber',
                'tag': hex(block_num),
                'boolean': 'true',
                'apikey': ETHERSCAN_API_KEY
            }
            
            response = requests.get(BASE_URL, params=params)
            if response.status_code == 200:
                block_data = response.json()
                if 'result' in block_data and 'transactions' in block_data['result']:
                    for tx in block_data['result']['transactions']:
                        unique_addresses.add(tx['from'])
                        if tx['to']:  # Some transactions (contract creations) may not have 'to'
                            unique_addresses.add(tx['to'])
                        
                        # If we have enough addresses, stop collecting
                        if len(unique_addresses) >= n:
                            break
            
            # If we have enough addresses, stop collecting
            if len(unique_addresses) >= n:
                break
        
        return list(unique_addresses)[:n]
    else:
        print(f"Failed to get current block: {response.status_code}")
        return []

def main():
    """Main function to fetch data and create wallet_data.csv"""
    # Prompt the user for wallet addresses
    print("Enter Ethereum wallet addresses separated by commas:")
    user_input = input().strip()
    addresses = [addr.strip() for addr in user_input.split(",") if addr.strip()]
    
    if not addresses:
        print("No valid addresses provided. Exiting.")
        return
    
    print(f"Processing {len(addresses)} addresses...")
    
    # Extract features for each address
    all_features = []
    for address in tqdm(addresses):
        features = extract_features(address)
        if features:
            all_features.append(features)
        
        # Respect API rate limits (5 calls/sec for free API key)
        time.sleep(1)
    
    if all_features:
        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_features)
        
        # Handle token name columns (convert to numeric)
        if ' ERC20_most_sent_token_type' in df.columns:
            # Map token names to numeric IDs
            token_dict = {token: idx for idx, token in enumerate(df[' ERC20_most_sent_token_type'].unique())}
            df[' ERC20_most_sent_token_type'] = df[' ERC20_most_sent_token_type'].map(token_dict)
        
        if ' ERC20_most_rec_token_type' in df.columns:
            token_dict = {token: idx for idx, token in enumerate(df[' ERC20_most_rec_token_type'].unique())}
            df[' ERC20_most_rec_token_type'] = df[' ERC20_most_rec_token_type'].map(token_dict)
        
        # Save DataFrame to a user-specific file
        output_file = "csv_files/wallet_data.csv"
        df.to_csv(output_file, index=True)
        print(f"Saved data for {len(df)} addresses to {output_file}")
    else:
        print("No valid address data collected.")

if __name__ == "__main__":
    main()