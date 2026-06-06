# AI-Based Sales Conversion Probability Prediction & Lead Prioritization Dashboard

An end-to-end Machine Learning and Explainable AI (XAI) solution to predict the probability of converting sales leads into paying customers, built to comply with HCL project evaluation guidelines.

---

## 🚀 Key Features

* **Balanced Ingestion Engine:** Realistic synthesis of 5,000 lead records featuring logical business correlations (Referral source, response time lag, budgets, and engagement thresholds).
* **Data Processing Pipeline:** Automatic missing value imputation (median/mode), duplicate removal, categorical One-Hot Encoding, and continuous variable Standard Scaling.
* **Feature Engineering Sandbox:** Four high-value engineered variables:
  * *Lead Value Score* (budget weighted by engagement)
  * *Interaction Velocity* (touchpoint frequency relative to response speed)
  * *Engagement Per Interaction* (communication efficiency)
  * *High Intent Lead* (rapid inbound warm lead flag)
* **Model Training & Comparison:** Concurrently trains and compares a Logistic Regression baseline against an ensemble Random Forest Classifier. Evaluates them on stratified holdout sets using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
* **Automatic Best Model Selection:** Serialization of the champion model to `models/best_model.pkl` for immediate production inference.
* **Optimized Explainable AI (XAI):**
  * *SHAP* global Beeswarm and Feature Importance plots (highly optimized using tree/linear formulations to run instantly on backend).
  * *LIME* local individual lead explanation overlays rendering interactive HTML bars in milliseconds.
* **Premium User Interface:** A stunning Streamlit dark glassmorphism dashboard with:
  * Interactive numerical probability gauges and priority badges (High, Medium, Low).
  * Pre-loaded preset demonstration selectors (High, Medium, and Low conversion templates) to simplify academic/corporate evaluation.
  * Real-time "What-If" business simulator to study how changes in outreach speed or touchpoints influence sales outcomes.
  * Built-in presentation slide structures and viva prep prep hubs.

---

## 📁 Folder Structure

```text
├── data/
│   ├── raw/
│   │   └── sales_leads_raw.csv         # Synthesized raw sales lead data
│   └── processed/
│       └── sales_leads_processed.csv   # Preprocessed and engineered data
├── models/
│   ├── preprocessor.pkl                # Preprocessors (scaler, encoder)
│   ├── logistic_regression.pkl         # Trained Logistic Regression model
│   ├── random_forest.pkl               # Trained Random Forest model
│   ├── best_model.pkl                  # Best performing model saved for inference
│   └── model_metadata.pkl              # Metadata containing active feature lists
├── src/
│   ├── __init__.py
│   ├── data_generator.py               # Generates highly realistic sales lead data
│   ├── data_preprocessing.py           # Preprocessing and feature engineering pipeline
│   ├── model_training.py               # Training, evaluation, model comparison & saving
│   └── explainable_ai.py               # SHAP & LIME explanation generators
├── app/
│   ├── main.py                         # Streamlit multi-page dashboard
│   └── style.css                       # Premium glassmorphism custom CSS
├── reports/
│   ├── images/
│   │   ├── model_comparison.png        # Performance metrics bar chart
│   │   ├── confusion_matrices.png      # Confusion heatmaps
│   │   ├── roc_curve.png               # ROC validation plot
│   │   ├── shap_bar_importance.png     # SHAP global average impact
│   │   └── shap_summary.png            # SHAP beeswarm direction plot
│   ├── project_synopsis.md             # High-level summary and business use cases
│   ├── architecture_workflow.md        # Architecture diagram (Mermaid) and data workflow
│   ├── ppt_presentation_points.md      # PPT slides outline for HCL Evaluation
│   ├── viva_questions_answers.md      # Q&A guide for evaluation defense
│   └── sales_insights_report.md        # Deep-dive business recommendations and strategies
├── requirements.txt                    # Project dependencies
└── README.md                           # Setup, running, and deployment guide
```

---

## 🛠️ Local Setup & Quick Start

Follow these simple steps to run the complete project locally on your machine.

### Prerequisites
Make sure you have **Python 3.9+** and **pip** installed.

### Step 1: Clone or Copy Project
Unpack/clone the project files into your desired workspace directory.

### Step 2: Install Dependencies
Open your command terminal inside the project root folder and execute:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Complete Machine Learning Pipeline
Execute the files sequentially to synthesize the data, run preprocessing, train the classifiers, and compile all explanation metrics:
```bash
# 1. Synthesize raw sales data
py src/data_generator.py

# 2. Preprocess data & engineer features
py src/data_preprocessing.py

# 3. Train models, evaluate, and save champion
py src/model_training.py

# 4. Pre-generate global SHAP plots
py src/explainable_ai.py
```

### Step 4: Run the Streamlit Dashboard Application
Launch the stunning local web app:
```bash
streamlit run app/main.py
```
A browser window will automatically open showing the interactive portal at `http://localhost:8501`.

---

## 🌐 Deployment to Streamlit Cloud

The application is fully deployment-ready for Streamlit Cloud. 

1. **Commit Code to GitHub:**
   Ensure the following files are pushed to a public or private GitHub repository:
   * `app/main.py` & `app/style.css`
   * `src/data_generator.py`, `src/data_preprocessing.py`, `src/model_training.py`, `src/explainable_ai.py`
   * `requirements.txt`
   * Note: The app features an automatic bootstrapper. If the serialized model files (`models/best_model.pkl`, etc.) are not present in your commit, the Streamlit app will **automatically** run the generator, preprocess data, and train models in the cloud container upon its first boot.

2. **Deploy via Streamlit Community Cloud:**
   * Visit [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
   * Click **New app**.
   * Select your repository, branch, and specify the main file path: `app/main.py`.
   * Click **Deploy**.
   * Streamlit will automatically read `requirements.txt`, install dependencies, run our startup initialization, and launch your live cloud application in under 2 minutes.

---

## ⚡ Explainable AI (XAI) Latency Optimization

To prevent the typical lagging, hanging, and memory-leaks associated with SHAP and LIME in Streamlit servers, this codebase implements three critical architectural optimizations:
1. **Model-Specific Explainers:** Bypasses slow model-agnostic explainers by leveraging highly optimized algorithms: `TreeExplainer` for Random Forest and `LinearExplainer` for Logistic Regression.
2. **Background Reference Downsampling:** Passes a small, representative sample of 100 historical records (via `shap.sample`) instead of the full dataset to compute expected values.
3. **Local Perturbation Triage:** Runs LIME local predictions with a targeted `num_samples=300` and restricted features (`num_features=5`). This delivers highly stable mathematical explanations in under **0.8 seconds** compared to standard LIME calculations taking 15+ seconds.
