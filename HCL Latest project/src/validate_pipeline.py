"""
Automated Pipeline Validation Script.
Author: Antigravity AI Coding Assistant
Description: Checks the integrity and existence of all required files,
             ensuring the complete ML pipeline ran and saved the correct assets.
"""

import os
import joblib

def validate_pipeline_assets():
    """
    Scans the project directory to verify all compiled assets and reports exist.
    """
    print("==================================================")
    print("SALES AI/ML PIPELINE: AUTOMATED INTEGRITY CHECK")
    print("==================================================\n")
    
    # 1. Check Directories
    dirs_to_check = ['data/raw', 'data/processed', 'models', 'reports/images', 'app']
    print("Checking project directory structures...")
    for d in dirs_to_check:
        if os.path.exists(d):
            print(f"  [PASS] Directory exists: '{d}'")
        else:
            print(f"  [FAIL] Directory missing: '{d}'")
            
    print("\nChecking generated data files...")
    # 2. Check Data
    raw_csv = 'data/raw/sales_leads_raw.csv'
    processed_csv = 'data/processed/sales_leads_processed.csv'
    
    for f in [raw_csv, processed_csv]:
        if os.path.exists(f):
            size_kb = os.path.getsize(f) / 1024
            print(f"  [PASS] File exists: '{f}' ({size_kb:.2f} KB)")
        else:
            print(f"  [FAIL] File missing: '{f}'")
            
    print("\nChecking trained serialized models...")
    # 3. Check Serialized Models
    models_to_check = [
        'models/preprocessor.pkl', 
        'models/logistic_regression.pkl', 
        'models/random_forest.pkl', 
        'models/best_model.pkl',
        'models/model_metadata.pkl'
    ]
    
    for m in models_to_check:
        if os.path.exists(m):
            print(f"  [PASS] Model exists: '{m}'")
        else:
            print(f"  [FAIL] Model missing: '{m}'")
            
    print("\nChecking evaluation visual plots...")
    # 4. Check Visualizations
    plots_to_check = [
        'reports/images/model_comparison.png',
        'reports/images/confusion_matrices.png',
        'reports/images/roc_curve.png'
    ]
    
    for p in plots_to_check:
        if os.path.exists(p):
            print(f"  [PASS] Plot exists: '{p}'")
        else:
            print(f"  [FAIL] Plot missing: '{p}'")
            
    # 5. Load and Verify Model Objects
    print("\nVerifying model objects load successfully...")
    try:
        model = joblib.load('models/best_model.pkl')
        metadata = joblib.load('models/model_metadata.pkl')
        print(f"  [PASS] Loaded best model successfully: Type is '{type(model).__name__}'")
        print(f"  [PASS] Champion Model Name in metadata: '{metadata['best_model_name']}'")
        print(f"  [PASS] Test F1-Score: {metadata['f1_score']:.4f} | ROC-AUC: {metadata['auc_score']:.4f}")
    except Exception as e:
        print(f"  [FAIL] Error loading model assets: {e}")
        
    print("\n==================================================")
    print("PIPELINE INTEGRITY CHECK COMPLETED")
    print("==================================================")

if __name__ == '__main__':
    validate_pipeline_assets()
