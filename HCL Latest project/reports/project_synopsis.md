# Project Synopsis: AI-Based Sales Conversion Probability Prediction

**Project Title:** AI-Based Sales Conversion Probability Prediction  
**Domain:** Predictive Sales Intelligence & Explainable AI (XAI)  
**Academic/Corporate Affiliation:** HCL Evaluation Framework  

---

## 1. Executive Summary
In today's highly competitive business landscape, sales teams are inundated with potential leads from multiple channels (websites, emails, social media, cold calling). Chasing every lead sequentially is highly inefficient, as sales representatives spend up to 70% of their time on cold prospects, leading to slow response times for warm prospects and lost revenue. 

This project delivers an end-to-end Machine Learning solution that predicts the probability of converting any sales lead into a paying customer. In addition to accuracy, this system integrates **Explainable AI (XAI)** frameworks (**SHAP** and **LIME**) so sales reps and business managers can immediately see the exact rationale behind each prediction. This allows sales teams to prioritize high-value leads and construct highly tailored negotiation strategies.

---

## 2. Project Objectives
* **Lead Scoring & Prioritization:** Quantify the conversion likelihood of each lead on a scale of 0% to 100% and categorize them into actionable priority tiers: **High, Medium, and Low**.
* **XAI Integration:** Explain model predictions transparently to build trust with business users and prevent "black-box" skepticism.
* **Modular Pipeline Development:** Create a clean, production-grade codebase encompassing data generation, preprocessing, training, auto-selection, explainability, and a modern frontend dashboard.
* **HCL Evaluation Compliance:** Meet all academic and engineering criteria, including detailed performance comparisons, robust feature engineering, and evaluation defense collaterals.

---

## 3. Technology Stack & Architecture
* **Language:** Python 3.9+
* **Data Manipulation & Engineering:** Pandas, NumPy
* **Machine Learning Algorithms:** Logistic Regression (parametric baseline), Random Forest Classifier (non-parametric ensemble)
* **Model Selection & Scaling:** Scikit-learn (StandardScaler, OneHotEncoder, TrainTestSplit)
* **Explainable AI:** SHAP (global/local attribution), LIME (local perturbation surrogate)
* **Dashboard Interface:** Streamlit (High-fidelity custom CSS with Glassmorphism)
* **Interactive Visualization:** Plotly, Seaborn, Matplotlib

---

## 4. Key Engineered Features
1. **Lead Value Score:** (Budget * Engagement Score) / 100 — Captures financial potential weighted by engagement.
2. **Interaction Velocity:** Interaction Count / (Response Time + 1) — Evaluates contact speed relative to touchpoints.
3. **Engagement Per Interaction:** Engagement Score / (Interaction Count + 1) — Reflects communication efficiency.
4. **High Intent Lead:** Binary flag representing extremely hot leads (Engagement > 75 and Response Time < 2 hours).

---

## 5. Main Deliverables
* **Trained ML Models:** Logistic Regression and Random Forest serialized models saved using `joblib` for inference.
* **Streamlit Web Application:** Interactive dashboard with real-time prediction gauges, preset demonstration selectors, custom LIME overlays, and business playbooks.
* **Global XAI Assets:** Saved SHAP summary plot and SHAP bar plot for structural lead evaluation.
* **HCL Project Documents:** Complete developer guide, architecture/workflow design, Viva prep cards, PPT outlines, and sales strategy reports.
