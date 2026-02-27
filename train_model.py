import pandas as pd
import numpy as np
import random
import lightgbm as lgb
import joblib

print("⚙️  Starting Real-World ML Model Training (LightGBM)...")

# --- 1. Profile Creation ---
def create_profiles(num_users=2000):
    users = []
    for i in range(num_users):
        users.append({
            'user_id': i, 'avg_tx': np.random.uniform(500, 4000),
            'std_dev_tx': np.random.uniform(100, 1500), 'max_tx': np.random.uniform(5000, 50000)
        })
    return users

# --- 2. Generate Messy, Realistic Dataset (Features match GUI) ---
def create_realistic_dataset(users, num_transactions=150000):
    data = []
    
    # 85% Clearly Legitimate
    for _ in range(int(num_transactions * 0.85)):
        user = random.choice(users)
        amount = abs(np.random.normal(user['avg_tx'], user['std_dev_tx']))
        data.append({
            'amount': amount, 'amount_to_avg_ratio': amount / user['avg_tx'],
            'is_new_beneficiary': 1 if random.random() < 0.2 else 0, # Normal new beneficiaries
            'is_new_device': 1 if random.random() < 0.05 else 0,
            'tx_velocity_1h': random.randint(1, 4), # Low velocity
            'is_collect_request': 0, 'is_scam': 0
        })

    # 5% Clearly Fraudulent
    for _ in range(int(num_transactions * 0.05)):
        user = random.choice(users)
        amount = random.choice([4999, 9999, user['max_tx'] * np.random.uniform(0.8, 0.99)])
        data.append({
            'amount': amount, 'amount_to_avg_ratio': amount / user['avg_tx'],
            'is_new_beneficiary': 1, # Scams are almost always to new beneficiaries
            'is_new_device': 1 if random.random() < 0.7 else 0,
            'tx_velocity_1h': random.randint(5, 15), # High velocity
            'is_collect_request': 1 if random.random() < 0.8 else 0, # Often collect
            'is_scam': 1
        })
        
    # 5% "Suspicious but Legit" (The Grey Area)
    for _ in range(int(num_transactions * 0.05)):
        user = random.choice(users)
        amount = user['max_tx'] * np.random.uniform(0.5, 0.9) 
        data.append({
            'amount': amount, 'amount_to_avg_ratio': amount / user['avg_tx'],
            'is_new_beneficiary': 1, # KEY: A legit tx to a new beneficiary
            'is_new_device': 1 if random.random() < 0.2 else 0,
            'tx_velocity_1h': random.randint(1, 3), # Low velocity
            'is_collect_request': 0, # Not a collect request
            'is_scam': 0 # Legit
        })

    # 5% "Low-key Scams" (The Grey Area)
    for _ in range(int(num_transactions * 0.05)):
        user = random.choice(users)
        amount = abs(np.random.normal(user['avg_tx'], user['std_dev_tx'] * 0.1))
        data.append({
            'amount': amount, 'amount_to_avg_ratio': amount / user['avg_tx'],
            'is_new_beneficiary': 1, # New beneficiary
            'is_new_device': 0, 'tx_velocity_1h': random.randint(1, 3), # Looks normal
            'is_collect_request': 1, # KEY: It's a collect request
            'is_scam': 1 # Scam
        })
        
    return pd.DataFrame(data).sample(frac=1).reset_index(drop=True)

# --- 3. Main Training Logic (LightGBM) ---
print("Creating advanced, 'messy' dataset with GUI-aligned features...")
users = create_profiles()
df = create_realistic_dataset(users)

print("Training the LightGBM model on the new dataset...")
X = df.drop('is_scam', axis=1)
y = df['is_scam']

lgbm_model = lgb.LGBMClassifier(
    objective='binary', metric='auc', n_estimators=500,
    learning_rate=0.05, is_unbalance=True, random_state=42, n_jobs=-1
)
lgbm_model.fit(X, y)

# --- 4. Save the New Model and Columns ---
joblib.dump(lgbm_model, 'lgbm_upi_model.joblib')
joblib.dump(X.columns, 'lgbm_model_columns.joblib')

print("\n✅ Real-world ML model (LGBM) and columns have been saved.")