# HCL Project Evaluation: Comprehensive Viva Prep Q&A

This document contains detailed, academic-level questions and answers designed to prepare you for your project defense, viva voce, or technical interview.

---

## Part 1: Project Objective & Dataset

### Q1: What is the core business problem this project addresses?
**Answer:** The project addresses the operational inefficiency in enterprise sales workflows. Inbound leads arrive from various channels, but sales reps waste up to 70% of their time calling cold prospects with zero intent to buy. Furthermore, delayed response times hurt conversion rates. This system predicts the conversion probability of each lead in real-time, categorizes them into actionable priority tiers (High, Medium, Low), and provides explanations using XAI. This helps sales teams prioritize high-value leads immediately, increasing sales productivity and conversion rates.

### Q2: Why did you create a synthetic dataset rather than use a public dataset?
**Answer:** Sales pipeline and customer lead data are highly proprietary, containing sensitive financial information (budgets), customer data, and sales call histories. Consequently, high-quality, realistic public datasets are rare. By designing a custom data generator (`src/data_generator.py`), we could:
1. Model realistic business interactions and non-linear relationships.
2. Control the noise ratio to simulate real-world data collection issues.
3. Incorporate specific features like "Engagement Score" and "Response Time" that are critical in modern digital sales but absent in generic CRM datasets.

---

## Part 2: Data Preprocessing & Feature Engineering

### Q3: How does your preprocessing pipeline handle missing values and duplicates?
**Answer:** The pipeline implements robust defensive cleaning in `src/data_preprocessing.py`:
* **Missing Values:** Numerical columns are imputed using the column median to protect against outliers. Categorical columns are imputed using the mode (most frequent value).
* **Duplicates:** The pipeline identifies and removes duplicate rows using pandas `.duplicated()` and `.drop_duplicates()`. This prevents data leakage and bias.

### Q4: Explain the feature engineering you performed. Why are these features valuable?
**Answer:** We engineered four new features to combine individual factors into high-level business signals:
1. **Lead Value Score:** `(Budget * Engagement_Score) / 100`. This scales the lead's financial capacity (Budget) by their interest level. A high budget lead with zero engagement is worth less attention than a moderate budget lead with peak engagement.
2. **Interaction Velocity:** `Interaction_Count / (Response_Time + 1)`. Measures how actively the sales team engages the lead relative to the response lag. Higher velocity signals high momentum in the sales cycle.
3. **Engagement Per Interaction:** `Engagement_Score / (Interaction_Count + 1)`. Measures the quality and responsiveness of each touchpoint. If engagement is high with few interactions, the lead is highly self-motivated.
4. **High Intent Lead:** A binary flag `((Engagement_Score > 75) & (Response_Time < 2.0))`. This flags hot inbound leads who clicked/downloaded materials and were contacted within 2 hours.

These engineered features help linear models capture non-linear and multiplicative interactions, boosting model accuracy and F1 scores.

### Q5: Why is scaling necessary, and how did you apply it?
**Answer:** Features like `Budget` (values up to $150,000) and `Response_Time` (values under 48 hours) exist on completely different scales. 
* **Logistic Regression:** Relies on gradient descent and regularization. Large scale differences cause gradient path oscillation and unfair penalization of coefficients.
* **Solution:** We applied `StandardScaler` from scikit-learn to all continuous numeric columns. This standardizes each feature to have a mean of 0 and a standard deviation of 1:
  $$z = \frac{x - \mu}{\sigma}$$
* The fitted scaler was saved as `models/preprocessor.pkl` to transform new, single-lead inference data in the same way during deployment.

---

## Part 3: Machine Learning Modeling & Comparison

### Q6: Why did you train both Logistic Regression and Random Forest?
**Answer:** We wanted to compare two fundamentally different families of ML algorithms:
1. **Logistic Regression:** A parametric, linear model. It is mathematically simple, extremely fast, highly interpretable, and serves as an excellent classification baseline.
2. **Random Forest Classifier:** A non-parametric, ensemble model composed of 100 decision trees. It inherently handles complex, non-linear interactions and categorical feature boundaries.

### Q7: Explain the metrics you used to compare your models.
**Answer:** We evaluated the models on a holdout test split (20%) using five core classification metrics:
* **Accuracy:** Overall proportion of correct predictions. (Can be misleading if classes are imbalanced).
* **Precision:** Out of all leads predicted to convert, how many actually converted? High precision prevents wasting sales time on false leads.
* **Recall (Sensitivity):** Out of all leads that actually converted, how many did the model catch? High recall ensures we don't miss warm prospects.
* **F1-Score:** The harmonic mean of Precision and Recall. It is the best metric for balanced optimization.
* **ROC-AUC:** Area Under the Receiver Operating Characteristic curve. Measures the model's ability to distinguish between classes across all decision thresholds.

### Q8: In your run, why did Logistic Regression perform as well or better than Random Forest?
**Answer:** In our training run, Logistic Regression achieved a slightly higher F1-score (0.7140 vs 0.7112) and ROC-AUC (0.7393 vs 0.7331). 
This occurs because our synthetic data generation process models conversion probabilities using a logit-based score (log-odds). Logistic Regression is mathematically designed to fit log-odds directly, making it highly stable. Additionally, Random Forest is prone to slight overfitting on complex training splits when deep decision boundaries are created, whereas Logistic Regression generalizes smoothly, showing that simpler, linear models are often highly robust and preferable in production.

---

## Part 4: Explainable AI (XAI)

### Q9: Why is Explainable AI (XAI) critical for this project?
**Answer:** In professional settings (such as enterprise sales, healthcare, and finance), users reject "black-box" models. A sales representative will not trust an AI that says "This lead has a 12% conversion chance" without explanation. Integrating SHAP and LIME:
1. **Builds Trust:** Explains the specific drivers behind predictions.
2. **Actionable Insights:** Tells the sales representative exactly *why* a lead is warm (e.g., fast response time) so they can reference it in their call.
3. **Regulatory Auditing:** Ensures predictions are fair, transparent, and comply with ethical AI requirements.

### Q10: How does SHAP calculate feature importance, and what is game theory's role?
**Answer:** SHAP is based on **Shapley values** from cooperative game theory. 
* **The Game:** The players are the individual feature values, and the payout is the model prediction.
* **Attribution:** To calculate a feature's importance fairly, SHAP measures how the model's prediction changes when that feature is present versus when it is absent. This difference is computed across all possible feature combinations (coalitions).
* **Properties:** This ensures consistency (features with higher impact always get higher SHAP values) and local accuracy (the sum of SHAP values equals the prediction minus the historical baseline average).

### Q11: What is the core difference between SHAP and LIME?
**Answer:** 
* **Scope:** SHAP provides both **global interpretability** (aggregate feature importance across the entire dataset) and **local interpretability** (explaining individual predictions). LIME is purely **local**.
* **Methodology:** SHAP is additive and calculates fair attribution across all coalitions using game theory. LIME works by generating a local surrogate model: it perturbs the input data point (adds noise), gets predictions for these perturbed points, weights them by distance to the original point, and fits a simple, interpretable linear model (like Ridge Regression) to outline the local boundary.
* **Performance:** LIME is highly flexible and model-agnostic, while SHAP is theoretically consistent but can be computationally expensive without optimizations.

### Q12: How did you optimize SHAP and LIME to run lag-free in Streamlit?
**Answer:**
1. **Explainer Choice:** Instead of using the model-agnostic `shap.KernelExplainer` (which takes minutes to run), we used optimized explainers: `shap.TreeExplainer` for Random Forest, and `shap.LinearExplainer` for Logistic Regression.
2. **Background Data Sampling:** We passed a small, representative sample of 100 background records (via `shap.sample(X, 100)`) instead of the entire 4,000-row training set to compute expected values.
3. **LIME Tuning:** We set LIME's perturbation size to `num_samples=300` and limited local explanations to the top `num_features=5`. This reduced LIME calculations from 15+ seconds down to under 0.8 seconds.
