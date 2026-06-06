"""
Data Preprocessing & Feature Engineering Module.
Author: Antigravity AI Coding Assistant
Description: This module cleans the raw lead data, implements feature engineering
             to boost prediction accuracy, and prepares variables for ML modeling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

def preprocess_data(input_csv_path, output_csv_dir=None, is_training=True, preprocessor_path=None):
    """
    Cleans raw sales lead data, engineers new features, and encodes/scales features.
    
    Parameters:
    -----------
    input_csv_path : str
        Path to the raw CSV dataset.
    output_csv_dir : str, optional
        Directory where processed CSV and models should be saved.
    is_training : bool
        If True, fits and saves the preprocessor. If False, loads saved preprocessor.
    preprocessor_path : str, optional
        Path to load/save the preprocessor joblib object.
        
    Returns:
    --------
    X : pd.DataFrame
        Processed features.
    y : pd.Series or None
        Target labels (if training or if column exists).
    df_engineered : pd.DataFrame
        Dataframe with newly engineered features, but before encoding/scaling (useful for EDA).
    """
    print(f"Loading raw data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # --- 1. HANDLE MISSING VALUES ---
    # Although our synthetic generator creates complete data, a real-world pipeline
    # must handle missing values. We fill numeric with median and categorical with mode.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in numeric column '{col}' with median: {median_val}")
            
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled missing values in categorical column '{col}' with mode: {mode_val}")
            
    # --- 2. REMOVE DUPLICATES ---
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Removed {num_duplicates} duplicate records.")
    else:
        print("No duplicate records found.")
        
    # --- 3. FEATURE ENGINEERING ---
    # We engineer four high-value business features:
    print("Engineering meaningful sales features...")
    
    # Feature 1: Lead Value Score (Budget multiplied by engagement percentage)
    df['Lead_Value_Score'] = (df['Budget'] * df['Engagement_Score']) / 100.0
    
    # Feature 2: Interaction Velocity (Touchpoints relative to response speed)
    # Adding 1 to response time prevents division by zero.
    df['Interaction_Velocity'] = df['Interaction_Count'] / (df['Response_Time'] + 1.0)
    
    # Feature 3: Engagement Per Interaction (Efficiency of communication)
    df['Engagement_Per_Interaction'] = df['Engagement_Score'] / (df['Interaction_Count'] + 1.0)
    
    # Feature 4: High Intent Lead (Binary flag: high engagement AND rapid sales follow-up)
    df['High_Intent_Lead'] = ((df['Engagement_Score'] > 75) & (df['Response_Time'] < 2.0)).astype(int)
    
    # Keep a copy of engineered but unencoded/unscaled dataframe for clean EDA/reporting
    df_engineered = df.copy()
    
    # --- 4. PREPARE FEATURES FOR MACHINE LEARNING ---
    # Drop identifier columns and the true probability (used only for generating target)
    cols_to_drop = ['Lead_ID']
    if 'Conversion_Probability_True' in df.columns:
        cols_to_drop.append('Conversion_Probability_True')
        
    df_features = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Extract target variable if it exists
    if 'Conversion_Status' in df_features.columns:
        y = df_features['Conversion_Status']
        X_raw = df_features.drop(columns=['Conversion_Status'])
    else:
        y = None
        X_raw = df_features
        
    # --- 5. CATEGORICAL ENCODING & CONTINUOUS SCALING ---
    # Numerical variables to scale
    numeric_features = [
        'Budget', 'Interaction_Count', 'Response_Time', 'Engagement_Score', 
        'Previous_Contact_History', 'Lead_Value_Score', 'Interaction_Velocity', 
        'Engagement_Per_Interaction'
    ]
    
    # Categorical variables to encode
    categorical_features = ['Lead_Source', 'Industry', 'Company_Size', 'Location']
    
    if is_training:
        print("Fitting scalers and encoders...")
        
        # Fit scaler on numeric features
        scaler = StandardScaler()
        X_scaled_numeric = scaler.fit_transform(X_raw[numeric_features])
        df_scaled_numeric = pd.DataFrame(X_scaled_numeric, columns=numeric_features, index=X_raw.index)
        
        # Fit encoder on categorical features
        # handle_unknown='ignore' ensures robustness for deployment
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded_categorical = encoder.fit_transform(X_raw[categorical_features])
        encoded_cols = encoder.get_feature_names_out(categorical_features)
        df_encoded_categorical = pd.DataFrame(X_encoded_categorical, columns=encoded_cols, index=X_raw.index)
        
        # Combine numerical and categorical features
        # High_Intent_Lead is already binary (0/1), so we keep it as-is without scaling/encoding
        X_processed = pd.concat([
            df_scaled_numeric, 
            df_encoded_categorical, 
            X_raw[['High_Intent_Lead']]
        ], axis=1)
        
        # Save preprocessors for deployment/inference
        if preprocessor_path:
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            joblib.dump({'scaler': scaler, 'encoder': encoder}, preprocessor_path)
            print(f"Saved preprocessing pipelines to: {preprocessor_path}")
            
    else:
        # INFERENCE/TEST MODE
        print(f"Loading preprocessors from: {preprocessor_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor object not found at {preprocessor_path}. Train the model first.")
            
        preprocessors = joblib.load(preprocessor_path)
        scaler = preprocessors['scaler']
        encoder = preprocessors['encoder']
        
        # Scale numeric features
        X_scaled_numeric = scaler.transform(X_raw[numeric_features])
        df_scaled_numeric = pd.DataFrame(X_scaled_numeric, columns=numeric_features, index=X_raw.index)
        
        # Encode categorical features
        X_encoded_categorical = encoder.transform(X_raw[categorical_features])
        encoded_cols = encoder.get_feature_names_out(categorical_features)
        df_encoded_categorical = pd.DataFrame(X_encoded_categorical, columns=encoded_cols, index=X_raw.index)
        
        # Combine
        X_processed = pd.concat([
            df_scaled_numeric, 
            df_encoded_categorical, 
            X_raw[['High_Intent_Lead']]
        ], axis=1)
        
    # --- 6. SAVE PROCESSED DATASET ---
    if output_csv_dir:
        os.makedirs(output_csv_dir, exist_ok=True)
        # For evaluation clarity, we'll save a file that is preprocessed for visual inspection
        processed_csv_path = os.path.join(output_csv_dir, 'sales_leads_processed.csv')
        df_engineered.to_csv(processed_csv_path, index=False)
        print(f"Saved engineered dataset (for EDA) to: {processed_csv_path}")
        
    return X_processed, y, df_engineered

if __name__ == '__main__':
    # Test execution
    input_path = os.path.join('data', 'raw', 'sales_leads_raw.csv')
    if os.path.exists(input_path):
        X, y, df_eng = preprocess_data(
            input_csv_path=input_path,
            output_csv_dir=os.path.join('data', 'processed'),
            is_training=True,
            preprocessor_path=os.path.join('models', 'preprocessor.pkl')
        )
        print(f"Preprocessing completed. Feature dimensions: {X.shape}")
    else:
        print("Raw dataset not found! Run data_generator.py first.")
