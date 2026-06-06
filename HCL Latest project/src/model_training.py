"""
Model Training, Evaluation & Auto-Selection Module.
Author: Antigravity AI Coding Assistant
Description: Trains Logistic Regression and Random Forest classifiers, evaluates
             them using comprehensive metrics, generates comparisons, and saves
             the best model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

def train_and_evaluate():
    """
    Trains and compares ML models on preprocessed lead data.
    Automatically saves the best model.
    """
    # Define directories
    processed_data_path = os.path.join('data', 'processed', 'sales_leads_processed.csv')
    preprocessor_pkl = os.path.join('models', 'preprocessor.pkl')
    
    # 1. Load Preprocessed Data
    # Since we need to re-encode/re-scale using the saved preprocessor, we'll import and run preprocess_data
    from data_preprocessing import preprocess_data
    
    raw_data_path = os.path.join('data', 'raw', 'sales_leads_raw.csv')
    X, y, df_eng = preprocess_data(
        input_csv_path=raw_data_path,
        is_training=False,
        preprocessor_path=preprocessor_pkl
    )
    
    print("\n--- Splitting Dataset into Train and Test Sets (80-20 Split) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training Features Shape: {X_train.shape}")
    print(f"Testing Features Shape:  {X_test.shape}")
    
    # 2. Train Models
    print("\n--- Training Logistic Regression Model ---")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    
    print("\n--- Training Random Forest Classifier Model ---")
    # n_estimators=100 and max_depth=12 provides excellent performance without overfitting
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Save both models for comparison later
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(log_reg, os.path.join(models_dir, 'logistic_regression.pkl'))
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
    
    # 3. Evaluate Models
    results = {}
    
    for name, model in [('Logistic Regression', log_reg), ('Random Forest', rf_model)]:
        # Predict labels
        y_pred = model.predict(X_test)
        # Predict probabilities
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate standard metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'Confusion_Matrix': cm,
            'Model_Object': model,
            'Y_Prob': y_prob
        }
        
        print(f"\n===== {name} Evaluation Results =====")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
    # 4. Generate Visualizations for Reports
    reports_img_dir = os.path.join('reports', 'images')
    os.makedirs(reports_img_dir, exist_ok=True)
    
    # Visualization A: Metrics Comparison Bar Chart
    df_metrics = pd.DataFrame({
        metric: [results['Logistic Regression'][metric], results['Random Forest'][metric]]
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    }, index=['Logistic Regression', 'Random Forest']).T
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Render grouped bar chart
    ax = df_metrics.plot(kind='bar', figsize=(10, 6), color=['#5A9', '#3F72AF'])
    plt.title('Machine Learning Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
    plt.xlabel('Evaluation Metrics', fontsize=12)
    plt.xticks(rotation=0)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right', frameon=True)
    
    # Annotate bar scores
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
                    
    plt.tight_layout()
    metrics_chart_path = os.path.join(reports_img_dir, 'model_comparison.png')
    plt.savefig(metrics_chart_path, dpi=300)
    plt.close()
    
    # Visualization B: Confusion Matrices Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, name in enumerate(['Logistic Regression', 'Random Forest']):
        cm = results[name]['Confusion_Matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    annot_kws={'size': 14, 'weight': 'bold'})
        axes[idx].set_title(f'{name} Confusion Matrix', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Predicted Conversion', fontsize=11)
        axes[idx].set_ylabel('Actual Conversion', fontsize=11)
        axes[idx].set_xticklabels(['Not Converted', 'Converted'])
        axes[idx].set_yticklabels(['Not Converted', 'Converted'])
        
    plt.tight_layout()
    cm_chart_path = os.path.join(reports_img_dir, 'confusion_matrices.png')
    plt.savefig(cm_chart_path, dpi=300)
    plt.close()
    
    # Visualization C: ROC Curves Plot
    plt.figure(figsize=(8, 6))
    for name, color in [('Logistic Regression', '#5A9'), ('Random Forest', '#3F72AF')]:
        y_prob = results[name]['Y_Prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = results[name]['ROC-AUC']
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc_score:.3f})')
        
    plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    roc_chart_path = os.path.join(reports_img_dir, 'roc_curve.png')
    plt.savefig(roc_chart_path, dpi=300)
    plt.close()
    
    # 5. Auto-Select the Best Model
    # We choose the model with the highest F1-Score on the test set
    best_model_name = max(results, key=lambda k: results[k]['F1-Score'])
    best_model = results[best_model_name]['Model_Object']
    best_f1 = results[best_model_name]['F1-Score']
    best_auc = results[best_model_name]['ROC-AUC']
    
    print(f"\n>>> AUTO-SELECTING THE BEST MODEL: <<<")
    print(f"Selected: {best_model_name} with F1-Score of {best_f1:.4f} and ROC-AUC of {best_auc:.4f}")
    
    # Save the best model
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(best_model, best_model_path)
    print(f"Saved the best model to: {best_model_path}")
    
    # Save metadata about which model is best
    metadata = {
        'best_model_name': best_model_name,
        'f1_score': best_f1,
        'auc_score': best_auc,
        'features_list': list(X.columns)
    }
    joblib.dump(metadata, os.path.join(models_dir, 'model_metadata.pkl'))
    
    print("Model training pipeline completed successfully!")

if __name__ == '__main__':
    train_and_evaluate()
