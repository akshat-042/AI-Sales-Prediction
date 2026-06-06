"""
Explainable AI (XAI) Helper Module.
Author: Antigravity AI Coding Assistant
Description: Implements high-performance SHAP and LIME explanations.
             Uses optimized explainers (TreeExplainer and LinearExplainer)
             and background sampling to avoid lagging during Streamlit execution.
"""

import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend to prevent GUI threads on server
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import joblib

def load_explainers_data():
    """
    Utility to load the dataset and best model for building explainers.
    """
    models_dir = 'models'
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    preprocessor_pkl = os.path.join(models_dir, 'preprocessor.pkl')
    metadata_pkl = os.path.join(models_dir, 'model_metadata.pkl')
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Models not trained yet. Run model_training.py first.")
        
    model = joblib.load(best_model_path)
    preprocessors = joblib.load(preprocessor_pkl)
    metadata = joblib.load(metadata_pkl)
    
    # Load training data to reconstruct feature structure
    raw_data_path = os.path.join('data', 'raw', 'sales_leads_raw.csv')
    from data_preprocessing import preprocess_data
    X_processed, y, _ = preprocess_data(
        input_csv_path=raw_data_path,
        is_training=False,
        preprocessor_path=preprocessor_pkl
    )
    
    return model, X_processed, y, preprocessors, metadata

def generate_global_shap_plots():
    """
    Generates and saves global SHAP summary and importance plots.
    These are pre-computed during training/validation to save rendering time in production.
    """
    print("\n--- Generating Global SHAP Explanations ---")
    model, X, y, _, metadata = load_explainers_data()
    best_model_name = metadata['best_model_name']
    
    # Select optimized explainer based on model architecture
    if 'Random Forest' in best_model_name:
        print("Using optimized TreeExplainer for Random Forest...")
        explainer = shap.TreeExplainer(model)
        # Random Forest yields shap values for each class [class_0, class_1]
        shap_values = explainer.shap_values(X)
        
        # In newer shap versions, shap_values might be a list or array.
        # For binary classification, shap_values[1] represents positive class impact
        if isinstance(shap_values, list):
            shap_values_pos = shap_values[1]
        elif len(shap_values.shape) == 3: # shape (samples, features, classes)
            shap_values_pos = shap_values[:, :, 1]
        else:
            shap_values_pos = shap_values
    else:
        print("Using optimized LinearExplainer for Logistic Regression...")
        # Sample background data for speed
        background = shap.sample(X, 100)
        explainer = shap.LinearExplainer(model, background)
        shap_values_pos = explainer.shap_values(X)
        
    # Save SHAP Summary Plot
    reports_img_dir = os.path.join('reports', 'images')
    os.makedirs(reports_img_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_pos, X, show=False)
    plt.title(f"SHAP Global Feature Impact ({best_model_name})", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    shap_summary_path = os.path.join(reports_img_dir, 'shap_summary.png')
    plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save SHAP Bar Importance Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap.Explanation(values=shap_values_pos, data=X, feature_names=X.columns), max_display=12, show=False)
    plt.title(f"SHAP Feature Importance ({best_model_name})", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    shap_bar_path = os.path.join(reports_img_dir, 'shap_bar_importance.png')
    plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Global SHAP plots saved to: {reports_img_dir}")
    return shap_summary_path, shap_bar_path

def get_lime_explainer_object(X_train):
    """
    Instantiates and returns a LimeTabularExplainer configured for high performance.
    """
    # LIME expects continuous scaled data representing the engineered feature matrix.
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=list(X_train.columns),
        class_names=['Not Converted', 'Converted'],
        mode='classification',
        kernel_width=3.0,
        random_state=42
    )
    return explainer

def get_single_lead_explanation_lime(model, X_train, lead_processed_df, num_features=6):
    """
    Generates a LIME explanation for a single, processed lead prediction.
    Highly optimized: uses limited features and samples to execute in <1 second.
    """
    explainer = get_lime_explainer_object(X_train)
    
    # Extract the lead feature values as a 1D numpy array
    lead_vector = lead_processed_df.iloc[0].values
    
    # Define prediction function wrapper depending on output dimension
    predict_fn = model.predict_proba
    
    # Generate explanation
    # num_samples=300 keeps the perturbation space small and execution instantaneous
    explanation = explainer.explain_instance(
        data_row=lead_vector,
        predict_fn=predict_fn,
        num_features=num_features,
        num_samples=300
    )
    
    return explanation

def get_single_lead_explanation_shap(model, X_train, lead_processed_df, best_model_name):
    """
    Generates a local SHAP explanation (force values/contributions) for a single lead.
    """
    if 'Random Forest' in best_model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(lead_processed_df)
        base_value = explainer.expected_value
        
        # Handle binary list structure or multi-dim array
        if isinstance(shap_values, list):
            shap_values_pos = shap_values[1][0]
            base_value_pos = base_value[1]
        elif len(shap_values.shape) == 3: # shape (samples, features, classes)
            shap_values_pos = shap_values[0, :, 1]
            base_value_pos = base_value[1]
        else:
            shap_values_pos = shap_values[0]
            base_value_pos = base_value
    else:
        background = shap.sample(X_train, 100)
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(lead_processed_df)
        base_value = explainer.expected_value
        
        shap_values_pos = shap_values[0]
        base_value_pos = base_value
        
    return shap_values_pos, base_value_pos

if __name__ == '__main__':
    # Test generation of global plots
    try:
        generate_global_shap_plots()
    except Exception as e:
        print(f"Error during SHAP plotting: {e}")
        print("Please ensure models are trained first.")
