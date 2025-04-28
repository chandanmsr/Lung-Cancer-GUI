import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_prepare(path="data/lcs.csv"):
    """Load and preprocess the lung cancer dataset."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')
    
    # Binary columns conversion
    binary_cols = [
        'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
        'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
        'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
        'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
    ]
    for c in binary_cols:
        df[c] = df[c].astype(str).str.strip() \
            .map({'2': 1, '1': 0, 'YES': 1, 'NO': 0}).astype(int)
    
    # Special columns
    df['GENDER'] = df['GENDER'].str.upper().map({'M': 1, 'F': 0}).astype(int)
    df['LUNG_CANCER'] = df['LUNG_CANCER'].str.upper() \
        .map({'YES': 1, 'NO': 0}).astype(int)

    # Split and scale data
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    # Train-test split with SMOTE
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y
    )
    Xtr, ytr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
    
    return df, Xtr, Xte, ytr, yte, X.columns.tolist(), scaler