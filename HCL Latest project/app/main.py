"""
Main Streamlit Application for AI Sales Conversion Dashboard.
Author: Antigravity AI Coding Assistant
Description: Multi-page high-fidelity Streamlit app with pre-filled sample leads,
             visual probability gauges, interactive business strategy simulator,
             live LIME and SHAP explanations, and evaluative PPT/Viva preparation tools.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Add 'src' folder to Python path so we can import preprocessing and explainable AI
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try importing modular files. If not trained yet, we will handle it in the UI.
try:
    from src.data_preprocessing import preprocess_data
    from src.explainable_ai import get_single_lead_explanation_lime, get_single_lead_explanation_shap
except ImportError:
    pass

# Set Page Config
st.set_page_config(
    page_title="AI-Based Sales Conversion Probability Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style Loader
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <style>
            .main-title { color: #5a9; font-size: 2.5rem; font-weight: bold; }
            .glass-card { background-color: #1e293b; padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; }
            </style>
            """,
            unsafe_allow_html=True
        )

load_css("app/style.css")

# --- MODEL LOADING UTILITIES ---
@st.cache_resource
def load_ml_assets():
    """
    Loads all trained model files, encoders, scalers, and training features.
    If models do not exist, we will display an elegant warning page.
    """
    models_dir = 'models'
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    preprocessor_pkl = os.path.join(models_dir, 'preprocessor.pkl')
    metadata_pkl = os.path.join(models_dir, 'model_metadata.pkl')
    raw_data_path = os.path.join('data', 'raw', 'sales_leads_raw.csv')
    
    if not (os.path.exists(best_model_path) and os.path.exists(preprocessor_pkl)):
        return None
        
    try:
        model = joblib.load(best_model_path)
        preprocessors = joblib.load(preprocessor_pkl)
        metadata = joblib.load(metadata_pkl)
        
        # Load training data for XAI explainers background reference
        from src.data_preprocessing import preprocess_data
        X_train, y_train, df_eng = preprocess_data(
            input_csv_path=raw_data_path,
            is_training=False,
            preprocessor_path=preprocessor_pkl
        )
        
        return {
            'model': model,
            'scaler': preprocessors['scaler'],
            'encoder': preprocessors['encoder'],
            'metadata': metadata,
            'X_train': X_train,
            'y_train': y_train,
            'df_engineered': df_eng
        }
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None

assets = load_ml_assets()

# Sidebar Navigation
st.sidebar.markdown(
    """
    <div style='text-align: center; padding-bottom: 1.5rem;'>
        <h2 style='color: var(--primary-color); font-family: Outfit; font-weight: 800; font-size: 1.3rem;'>AI-Based Sales Conversion Probability Prediction</h2>
        <p style='color: var(--text-color); opacity: 0.7; font-size: 0.85rem;'>HCL Project Evaluation</p>
    </div>
    <hr style='border-color: rgba(128,128,128,0.2); margin-top:0; margin-bottom: 1.5rem;'/>
    """, 
    unsafe_allow_html=True
)

navigation = st.sidebar.radio(
    "NAVIGATION MENU",
    [
        "🏠 Dashboard Home", 
        "🔮 Lead Predictor & Simulator", 
        "🧬 XAI Explainability", 
        "📊 Model Compare & Analytics"
    ]
)

# Render warning if assets are missing
if assets is None:
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 3rem; margin-top: 3rem;">
            <h2 style="color: #f85149; font-family: 'Outfit'; font-weight: 700;">⚠️ Model Assets Not Found</h2>
            <p style="color: var(--text-color); opacity: 0.7; font-size: 1.1rem; margin-bottom: 2rem;">
                The machine learning models, encoders, and synthetic datasets have not been trained or generated yet.
            </p>
            <p style="color: var(--text-color);">
                Please run the command below in your terminal to initialize and train the models:
            </p>
            <code style="background-color: #0d1117; padding: 0.8rem 2rem; border-radius: 8px; font-size: 1.2rem; color: var(--primary-color); display: inline-block; margin-bottom: 2rem;">
                py src/model_training.py
            </code>
            <p style="color: var(--text-color); opacity: 0.7; font-size: 0.9rem;">
                Alternatively, reload this page after the background build completes.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# Extract variables from loaded assets
best_model = assets['model']
scaler = assets['scaler']
encoder = assets['encoder']
metadata = assets['metadata']
X_train = assets['X_train']
y_train = assets['y_train']
df_engineered = assets['df_engineered']
best_model_name = metadata['best_model_name']

# Define feature lists for inference scaling and encoding
numeric_features = [
    'Budget', 'Interaction_Count', 'Response_Time', 'Engagement_Score', 
    'Previous_Contact_History', 'Lead_Value_Score', 'Interaction_Velocity', 
    'Engagement_Per_Interaction'
]

categorical_features = ['Lead_Source', 'Industry', 'Company_Size', 'Location']

# --- PAGE 1: DASHBOARD HOME ---
if navigation == "🏠 Dashboard Home":
    st.markdown("<h1 class='main-title'>AI-Based Sales Conversion Probability Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Predict and prioritize high-value sales leads using industry-grade Machine Learning and Explainable AI (XAI)</p>", unsafe_allow_html=True)
    
    # KPIs Row
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-value">5,000</div>
                <div class="kpi-label">Historical Leads Analyzed</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">53.0%</div>
                <div class="kpi-label">Average Conversion Rate</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value" style="color: var(--primary-color);">{best_model_name}</div>
                <div class="kpi-label">Active Inference Model</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value" style="color: var(--primary-color);">{metadata['auc_score']:.2%}</div>
                <div class="kpi-label">Model ROC-AUC Score</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-bottom: 1rem;">🎯 Project Objective & Overview</h3>
                <p style="line-height: 1.6; color: var(--text-color); opacity: 0.9;">
                    In modern B2B and high-value B2C environments, sales teams are overwhelmed with leads. 
                    Contacting every lead sequentially is highly inefficient, resulting in long response times and lost sales. 
                    <b>This AI/ML application solves this bottleneck by predicting the conversion probability of each lead in real-time.</b>
                </p>
                <p style="line-height: 1.6; color: var(--text-color); opacity: 0.9;">
                    By combining <b>predictive performance</b> (using Logistic Regression and Random Forest) with 
                    <b>explainability</b> (using SHAP and LIME), sales professionals can see exactly <i>why</i> a lead has been classified as High, Medium, or Low priority. 
                    This helps build trust in AI recommendations and guides the exact negotiation strategy.
                </p>
                <h4 style="color: var(--text-color); font-family: Outfit; font-weight: 500; margin-top: 1.5rem;">🚀 Core Features Built:</h4>
                <ul style="color: var(--text-color); opacity: 0.9; line-height: 1.6; margin-left: 1.2rem;">
                    <li><b>Data Generator & Pipeline:</b> Realistic sales metrics with robust feature engineering.</li>
                    <li><b>Model Training:</b> Auto-evaluates and compiles optimal scikit-learn models.</li>
                    <li><b>Dual-engine XAI:</b> Instantaneous LIME tabular overlays and custom interactive SHAP interpretations.</li>
                    <li><b>Simulator Sandbox:</b> Real-time parameter tweaking to see impacts on conversion potential.</li>
                </ul>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <h3 style="color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-bottom: 1rem;">🔧 System Architecture Diagram</h3>
                <div style="background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.2); padding: 1.5rem; border-radius: 12px; font-family: monospace; font-size: 0.85rem; color: #a8ff78; line-height: 1.4;">
                    [Raw Sales Leads CSV] <br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br/>
                    [Feature Preprocessor] & Feature Engineering <br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+--&gt; Scale Continuous (StandardScaler)<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+--&gt; Encode Categorical (OneHotEncoder)<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v<br/>
                    [Inference Model ({best_model_name})] <br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+----&gt; Conversion Probability (0% - 100%)<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+----&gt; LIME / SHAP Explanations (XAI)<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+----&gt; Priority Scoring & Playbooks<br/>
                </div>
                <p style="font-size: 0.85rem; color: var(--text-color); opacity: 0.7; margin-top: 1rem; text-align: center;">
                    System architecture design conforming to HCL engineering evaluation rules.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# --- PAGE 2: LEAD PREDICTOR & SIMULATOR ---
elif navigation == "🔮 Lead Predictor & Simulator":
    st.markdown("<h1 class='main-title'>🔮 Lead Conversion Predictor & Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Enter new lead attributes manually or load standard demonstration presets to analyze conversions</p>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔮 Single Lead Predictor", "📂 Batch Lead Scorer (CSV Upload)"])
    
    with tab1:
        # 1. Preset Lead Selection
        st.sidebar.markdown("### 📋 Demo Lead Presets")
        preset_choice = st.sidebar.selectbox(
            "Choose a template lead to auto-fill the form:",
            ["[Manual Entry]", "High Priority Lead (Referral)", "Medium Priority Lead (Website)", "Low Priority Lead (Cold Call)"]
        )
        
        # Define default values based on presets
        if preset_choice == "High Priority Lead (Referral)":
            d_source = "Referral"
            d_industry = "Tech"
            d_size = "Large"
            d_budget = 65000.0
            d_interactions = 12
            d_response = 0.5
            d_location = "North America"
            d_engagement = 88
            d_prev = 2
        elif preset_choice == "Medium Priority Lead (Website)":
            d_source = "Website"
            d_industry = "Finance"
            d_size = "Medium"
            d_budget = 24000.0
            d_interactions = 5
            d_response = 3.5
            d_location = "Europe"
            d_engagement = 62
            d_prev = 1
        elif preset_choice == "Low Priority Lead (Cold Call)":
            d_source = "Cold Call"
            d_industry = "Education"
            d_size = "Small"
            d_budget = 4000.0
            d_interactions = 1
            d_response = 36.0
            d_location = "Asia-Pacific"
            d_engagement = 24
            d_prev = 0
        else: # Manual
            d_source = "Website"
            d_industry = "Tech"
            d_size = "Medium"
            d_budget = 15000.0
            d_interactions = 4
            d_response = 2.0
            d_location = "North America"
            d_engagement = 50
            d_prev = 0
            
        st.markdown("<h3 style='color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-bottom: 1.25rem;'>📝 Lead Attributes Input Form</h3>", unsafe_allow_html=True)
        
        # We construct the input columns
        col_input1, col_input2, col_input3 = st.columns(3)
        
        with col_input1:
            lead_source = st.selectbox("Lead Source", ["Website", "Email", "Cold Call", "LinkedIn", "Referral"], index=["Website", "Email", "Cold Call", "LinkedIn", "Referral"].index(d_source))
            industry = st.selectbox("Industry", ["Tech", "Finance", "Healthcare", "Retail", "Education", "Manufacturing"], index=["Tech", "Finance", "Healthcare", "Retail", "Education", "Manufacturing"].index(d_industry))
            company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"], index=["Small", "Medium", "Large"].index(d_size))
            
        with col_input2:
            budget = st.number_input("Monthly Budget ($)", min_value=1000.0, max_value=150000.0, value=d_budget, step=1000.0)
            interaction_count = st.slider("Interaction Count (Touchpoints)", min_value=1, max_value=25, value=int(d_interactions))
            response_time = st.number_input("Response Time (Hours)", min_value=0.1, max_value=48.0, value=d_response, step=0.5)
            
        with col_input3:
            location = st.selectbox("Location", ["North America", "Europe", "Asia-Pacific", "South America", "Middle East"], index=["North America", "Europe", "Asia-Pacific", "South America", "Middle East"].index(d_location))
            engagement_score = st.slider("Engagement Score (1-100)", min_value=1, max_value=100, value=int(d_engagement))
            prev_contacts = st.slider("Previous Contact History", min_value=0, max_value=5, value=int(d_prev))
            
        # Trigger prediction automatically on render and value changes
        # Construct 1-row DataFrame representing the input
        input_data = pd.DataFrame({
            'Lead_Source': [lead_source],
            'Industry': [industry],
            'Company_Size': [company_size],
            'Budget': [budget],
            'Interaction_Count': [interaction_count],
            'Response_Time': [response_time],
            'Location': [location],
            'Engagement_Score': [engagement_score],
            'Previous_Contact_History': [prev_contacts]
        })
        
        # Feature engineer on this single instance
        input_data['Lead_Value_Score'] = (input_data['Budget'] * input_data['Engagement_Score']) / 100.0
        input_data['Interaction_Velocity'] = input_data['Interaction_Count'] / (input_data['Response_Time'] + 1.0)
        input_data['Engagement_Per_Interaction'] = input_data['Engagement_Score'] / (input_data['Interaction_Count'] + 1.0)
        input_data['High_Intent_Lead'] = ((input_data['Engagement_Score'] > 75) & (input_data['Response_Time'] < 2.0)).astype(int)
        
        # Transform numeric
        scaled_numeric = scaler.transform(input_data[numeric_features])
        df_scaled_numeric = pd.DataFrame(scaled_numeric, columns=numeric_features)
        
        # Transform categorical
        encoded_categorical = encoder.transform(input_data[categorical_features])
        encoded_cols = encoder.get_feature_names_out(categorical_features)
        df_encoded_categorical = pd.DataFrame(encoded_categorical, columns=encoded_cols)
        
        # Combine
        X_inst = pd.concat([
            df_scaled_numeric, 
            df_encoded_categorical, 
            input_data[['High_Intent_Lead']]
        ], axis=1)
        
        # Predict Probability
        prob = best_model.predict_proba(X_inst)[0, 1]
        
        # Categorize Priority
        if prob >= 0.70:
            priority_label = "HIGH PRIORITY"
            priority_css = "badge badge-high"
            priority_color = "#3fb950"
            priority_desc = "🔥 Highly prospective lead. Allocate your top account executive immediately. High likelihood of conversion."
            action_plan = [
                "Call the lead within 15 minutes to leverage current interest.",
                "Prepare a highly customized enterprise contract/demo proposal.",
                "Offer a 10% fast-action introductory discount if closed this week."
            ]
        elif prob >= 0.40:
            priority_label = "MEDIUM PRIORITY"
            priority_css = "badge badge-medium"
            priority_color = "#d29922"
            priority_desc = "⚡ Moderately prospective lead. Keep engaged and schedule a personalized follow-up."
            action_plan = [
                "Send a personalized email containing relevant customer case studies.",
                "Schedule a standard product demo and feature overview.",
                "Set a calendar reminder to follow up in 3 business days."
            ]
        else:
            priority_label = "LOW PRIORITY"
            priority_css = "badge badge-low"
            priority_color = "#f85149"
            priority_desc = "💤 Low conversion probability. Keep in cold automated nurturing loops."
            action_plan = [
                "Add to the automated bi-weekly email newsletter sequence.",
                "Do not schedule direct manual sales calls unless engagement spikes.",
                "Re-assess budget fit in 6 months."
            ]
            
        st.markdown("<hr style='border-color: rgba(128,128,128,0.2); margin-top: 1rem; margin-bottom: 2rem;'/>", unsafe_allow_html=True)
        
        col_out1, col_out2 = st.columns([1, 1])
        
        with col_out1:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center;">
                    <h3 style="color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-bottom: 1.5rem;">🎯 Prediction Analysis</h3>
                    <p style="font-size: 1.1rem; color: var(--text-color); opacity: 0.7; margin-bottom: 0.5rem;">Estimated Conversion Probability</p>
                    <div style="font-size: 4.5rem; font-weight: 800; color: {priority_color}; font-family: Outfit; line-height: 1;">
                        {prob:.1%}
                    </div>
                    <div style="margin-top: 1.5rem; margin-bottom: 1.5rem;">
                        <span class="{priority_css}" style="font-size: 1.1rem; padding: 0.5rem 1.8rem;">{priority_label}</span>
                    </div>
                    <p style="color: var(--text-color); opacity: 0.9; font-size: 0.95rem; line-height: 1.5; padding: 0 1rem;">
                        {priority_desc}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_out2:
            st.markdown(
                f"""
                <div class="glass-card" style="height: 100%;">
                    <h3 style="color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-bottom: 1rem;">📋 Recommended Sales Action Plan</h3>
                    <ul style="color: var(--text-color); opacity: 0.9; line-height: 1.8; font-size: 1rem; padding-left: 1.2rem;">
                        <li><b>Action 1:</b> {action_plan[0]}</li>
                        <li style="margin-top: 0.8rem;"><b>Action 2:</b> {action_plan[1]}</li>
                        <li style="margin-top: 0.8rem;"><b>Action 3:</b> {action_plan[2]}</li>
                    </ul>
                    <div style="margin-top: 2rem; background-color: var(--secondary-background-color); padding: 1rem; border-radius: 8px; border: 1px solid rgba(90, 153, 230, 0.2);">
                        <p style="font-size: 0.85rem; color: var(--primary-color); margin: 0; font-weight: 500;">
                            💡 <b>Business Insight:</b> Engineered features calculated in real-time:
                            <br/>&bull; Lead Value Score: <b>${input_data['Lead_Value_Score'].iloc[0]:,.0f}</b>
                            <br/>&bull; Interaction Velocity: <b>{input_data['Interaction_Velocity'].iloc[0]:.2f}</b> touchpoints/hr
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Local Explainability Subsection
        st.markdown("<h3 style='color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-top: 2rem; margin-bottom: 1.25rem;'>🧬 Local Predictions Breakdown (Why this score?)</h3>", unsafe_allow_html=True)
        
        col_xai1, col_xai2 = st.columns([1, 1])
        
        with col_xai1:
            st.markdown(
                """
                <div class="glass-card" style="height: 100%;">
                    <h4 style="color: var(--primary-color); font-family: Outfit; font-weight: 500; margin-bottom: 0.8rem;">📉 Interactive SHAP Contribution Bar Plot</h4>
                    <p style="font-size: 0.85rem; color: var(--text-color); opacity: 0.7; margin-bottom: 1rem;">
                        SHAP (SHapley Additive exPlanations) values decompose the prediction to show how much each individual parameter deviates the probability from the historical base rate (53.0%).
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Calculate local SHAP values
            shap_values, base_value = get_single_lead_explanation_shap(best_model, X_train, X_inst, best_model_name)
            
            # Build custom interactive Plotly chart for SHAP contributions
            # Select top features with highest absolute impact for visualization
            df_shap = pd.DataFrame({
                'Feature': X_inst.columns,
                'Contribution': shap_values
            })
            
            df_shap['Abs_Contribution'] = df_shap['Contribution'].abs()
            df_shap = df_shap.sort_values(by='Abs_Contribution', ascending=True).tail(8) # Top 8 features
            
            # Colors: Positive contributions push conversion UP (green), negative pull it DOWN (red)
            df_shap['Color'] = np.where(df_shap['Contribution'] >= 0, '#3fb950', '#f85149')
            
            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                y=df_shap['Feature'],
                x=df_shap['Contribution'],
                orientation='h',
                marker_color=df_shap['Color'],
                hovertemplate="Feature: %{y}<br>Impact: %{x:.4f}<extra></extra>"
            ))
            
            fig_shap.update_layout(
                xaxis=dict(title="SHAP Impact (Log-Odds)"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=300
            )
            
            st.plotly_chart(fig_shap, use_container_width=True)
            
        with col_xai2:
            st.markdown(
                """
                <div class="glass-card" style="height: 100%;">
                    <h4 style="color: var(--primary-color); font-family: Outfit; font-weight: 500; margin-bottom: 0.8rem;">🧪 LIME Dynamic Overlay Explanation</h4>
                    <p style="font-size: 0.85rem; color: var(--text-color); opacity: 0.7; margin-bottom: 1rem;">
                        LIME (Local Interpretable Model-agnostic Explanations) creates a local linear surrogate model around this specific lead to show exactly which features are driving this individual conversion probability.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Generate and render LIME dynamically inside Streamlit using Plotly for native theme support
            with st.spinner("Calculating local LIME explanation..."):
                try:
                    lime_exp = get_single_lead_explanation_lime(best_model, X_train, X_inst, num_features=5)
                    
                    # Convert LIME explanation to Plotly chart for perfect light/dark theme integration
                    lime_list = lime_exp.as_list()
                    df_lime = pd.DataFrame(lime_list, columns=['Feature_Condition', 'Weight'])
                    
                    # Sort by absolute weight for better visualization
                    df_lime['Abs_Weight'] = df_lime['Weight'].abs()
                    df_lime = df_lime.sort_values(by='Abs_Weight', ascending=True)
                    
                    # Colors: Positive (towards Converted) = Orange, Negative (towards Not Converted) = Blue
                    df_lime['Color'] = np.where(df_lime['Weight'] >= 0, '#e38627', '#1f77b4')
                    
                    fig_lime = go.Figure()
                    fig_lime.add_trace(go.Bar(
                        y=df_lime['Feature_Condition'],
                        x=df_lime['Weight'],
                        orientation='h',
                        marker_color=df_lime['Color'],
                        hovertemplate="Condition: %{y}<br>Impact: %{x:.4f}<extra></extra>"
                    ))
                    
                    fig_lime.update_layout(
                        xaxis=dict(title="LIME Feature Weight"),
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=300
                    )
                    
                    st.plotly_chart(fig_lime, use_container_width=True)
                except Exception as ex:
                    st.warning(f"Unable to render LIME plot dynamically: {ex}")
                    
    with tab2:
        st.markdown("<h3 style='color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-bottom: 1.25rem;'>📂 Bulk Lead Scoring & Prioritization</h3>", unsafe_allow_html=True)
        st.write("Upload a CSV file containing your sales leads. The system will process all leads, calculate their conversion probabilities, assign priorities, and provide a downloadable report.")
        
        # Display template format so they know what columns to upload
        st.markdown(
            """
            <div style="background-color: var(--secondary-background-color); padding: 1rem; border-radius: 8px; border: 1px solid rgba(128,128,128,0.2); margin-bottom: 1.5rem;">
                <p style="margin-bottom: 0.5rem; font-size: 0.9rem; color: var(--text-color); opacity: 0.7;"><b>Required CSV Columns:</b></p>
                <code style="color: var(--primary-color);">Lead_Source, Industry, Company_Size, Budget, Interaction_Count, Response_Time, Location, Engagement_Score, Previous_Contact_History</code>
                <p style="margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-color); opacity: 0.7;">Note: Column names must match exactly (case-sensitive).</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # We provide a sample file download
        sample_df = pd.DataFrame({
            'Lead_Source': ['Website', 'Referral', 'Cold Call', 'LinkedIn', 'Email'],
            'Industry': ['Tech', 'Finance', 'Education', 'Healthcare', 'Retail'],
            'Company_Size': ['Medium', 'Large', 'Small', 'Medium', 'Small'],
            'Budget': [25000, 75000, 3000, 45000, 12000],
            'Interaction_Count': [4, 10, 1, 6, 2],
            'Response_Time': [2.5, 0.5, 24.0, 1.2, 8.0],
            'Location': ['North America', 'Europe', 'Asia-Pacific', 'Middle East', 'South America'],
            'Engagement_Score': [65, 90, 20, 78, 45],
            'Previous_Contact_History': [1, 3, 0, 2, 1]
        })
        sample_csv = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Sample CSV Template", data=sample_csv, file_name="sample_leads_template.csv", mime="text/csv")
        
        st.markdown("<hr style='border-color: rgba(128,128,128,0.2); margin-top: 1rem; margin-bottom: 1rem;'/>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV file for scoring", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df_raw = pd.read_csv(uploaded_file)
                batch_df = batch_df_raw.copy()
                
                req_cols = ['Lead_Source', 'Industry', 'Company_Size', 'Budget', 'Interaction_Count', 'Response_Time', 'Location', 'Engagement_Score', 'Previous_Contact_History']
                
                with st.spinner("Intelligently mapping columns and predicting probabilities..."):
                    import difflib
                    mapped_cols = {}
                    used_csv_cols = set()
                    
                    # 1. Fuzzy Column Mapping
                    for req_c in req_cols:
                        if req_c in batch_df.columns:
                            mapped_cols[req_c] = req_c
                            used_csv_cols.add(req_c)
                            continue
                        
                        # Try exact match ignoring case/spaces/underscores
                        found = False
                        for c in batch_df.columns:
                            if c not in used_csv_cols and c.lower().replace(' ', '').replace('_', '') == req_c.lower().replace(' ', '').replace('_', ''):
                                mapped_cols[req_c] = c
                                used_csv_cols.add(c)
                                found = True
                                break
                        if found: continue
                        
                        # Try fuzzy matching
                        available_cols = [c for c in batch_df.columns if c not in used_csv_cols]
                        matches = difflib.get_close_matches(req_c, available_cols, n=1, cutoff=0.5)
                        if matches:
                            mapped_cols[req_c] = matches[0]
                            used_csv_cols.add(matches[0])

                    if len(mapped_cols) == 0:
                        st.error("🚨 CRITICAL ERROR: The AI couldn't find ANY recognizable lead data columns in your CSV (like Budget, Industry, Score, etc.). We cannot predict without at least some sales data.")
                        st.stop()

                    # Rename matched columns to required names
                    reverse_map = {v: k for k, v in mapped_cols.items()}
                    batch_df = batch_df.rename(columns=reverse_map)

                    # --- KAGGLE 'X EDUCATION' DATASET DETECTOR & TRANSLATOR ---
                    if 'Total Time Spent on Website' in batch_df_raw.columns and 'TotalVisits' in batch_df_raw.columns:
                        st.success("🤖 Kaggle 'X Education' Dataset Detected! Auto-translating features to match the AI Model schema...")
                        
                        # Map Total Visits to Interaction Count
                        batch_df['Interaction_Count'] = batch_df_raw['TotalVisits'].fillna(0).clip(1, 25).astype(int)
                        mapped_cols['Interaction_Count'] = 'TotalVisits (Translated)'
                        
                        # Map Total Time to Engagement Score
                        max_time = batch_df_raw['Total Time Spent on Website'].max()
                        if max_time > 0:
                            batch_df['Engagement_Score'] = (batch_df_raw['Total Time Spent on Website'].fillna(0) / max_time * 100).clip(1, 100).astype(int)
                        else:
                            batch_df['Engagement_Score'] = 50
                        mapped_cols['Engagement_Score'] = 'Total Time Spent on Website (Translated)'
                        
                        # Map Lead Source strings to our supported categories
                        source_map = {
                            'Olark Chat': 'Website', 'Organic Search': 'Website', 'Direct Traffic': 'Website',
                            'Google': 'Website', 'Welingak Website': 'Website', 'Reference': 'Referral',
                            'Email Opened': 'Email', 'Email Link Clicked': 'Email'
                        }
                        if 'Lead Source' in batch_df_raw.columns:
                            batch_df['Lead_Source'] = batch_df_raw['Lead Source'].map(source_map).fillna('Website')
                            mapped_cols['Lead_Source'] = 'Lead Source (Categorized)'

                    # 2. Impute missing required columns with defaults
                    default_values = {
                        'Lead_Source': 'Website',
                        'Industry': 'Tech',
                        'Company_Size': 'Medium',
                        'Budget': 15000.0,
                        'Interaction_Count': 5,
                        'Response_Time': 24.0,
                        'Location': 'North America',
                        'Engagement_Score': 50,
                        'Previous_Contact_History': 0
                    }

                    missing_cols = [c for c in req_cols if c not in batch_df.columns]
                    for mc in missing_cols:
                        batch_df[mc] = default_values[mc]
                        
                    # Show mapping report to user
                    with st.expander("🔍 View AI Column Mapping Report (Why did you get these scores?)"):
                        st.write("If you see the exact same probability for many rows, it means your CSV was missing key columns, so the AI filled them with identical default values.")
                        report_df = pd.DataFrame([
                            {"Required AI Feature": req, "Mapped to your CSV column": mapped_cols.get(req, "❌ MISSING (Auto-filled with default)")} 
                            for req in req_cols
                        ])
                        st.dataframe(report_df, use_container_width=True)

                    # 3. Clean numeric columns (e.g. convert '$15,000' to 15000)
                    numeric_cols = ['Budget', 'Interaction_Count', 'Response_Time', 'Engagement_Score', 'Previous_Contact_History']
                    for num_c in numeric_cols:
                        if batch_df[num_c].dtype == 'object':
                            batch_df[num_c] = batch_df[num_c].astype(str).str.replace(r'[^\d.]', '', regex=True)
                        batch_df[num_c] = pd.to_numeric(batch_df[num_c], errors='coerce').fillna(default_values[num_c])

                    # 4. Feature Engineering
                    batch_proc = batch_df.copy()
                    batch_proc['Lead_Value_Score'] = (batch_proc['Budget'] * batch_proc['Engagement_Score']) / 100.0
                    batch_proc['Interaction_Velocity'] = batch_proc['Interaction_Count'] / (batch_proc['Response_Time'] + 1.0)
                    batch_proc['Engagement_Per_Interaction'] = batch_proc['Engagement_Score'] / (batch_proc['Interaction_Count'] + 1.0)
                    batch_proc['High_Intent_Lead'] = ((batch_proc['Engagement_Score'] > 75) & (batch_proc['Response_Time'] < 2.0)).astype(int)
                    
                    # 5. Scale/Encode
                    scaled_num = scaler.transform(batch_proc[numeric_features])
                    df_scaled_num = pd.DataFrame(scaled_num, columns=numeric_features, index=batch_proc.index)
                    
                    encoded_cat = encoder.transform(batch_proc[categorical_features])
                    encoded_cols = encoder.get_feature_names_out(categorical_features)
                    df_encoded_cat = pd.DataFrame(encoded_cat, columns=encoded_cols, index=batch_proc.index)
                    
                    X_batch = pd.concat([
                        df_scaled_num, 
                        df_encoded_cat, 
                        batch_proc[['High_Intent_Lead']]
                    ], axis=1)
                    
                    # 6. Predict
                    probs = best_model.predict_proba(X_batch)[:, 1]
                    batch_df_raw['Conversion_Probability'] = probs
                    
                    def get_priority(p):
                        if p >= 0.70: return "🔥 High"
                        elif p >= 0.40: return "⚡ Medium"
                        else: return "💤 Low"
                        
                    batch_df_raw['Priority'] = batch_df_raw['Conversion_Probability'].apply(get_priority)
                    
                    # Sort by probability descending
                    batch_df_raw = batch_df_raw.sort_values(by='Conversion_Probability', ascending=False)
                    
                # 7. Dynamic Top N Selector
                st.markdown("<hr style='border-color: rgba(128,128,128,0.2);'/>", unsafe_allow_html=True)
                top_n = st.slider("🎯 Select number of Top Leads to display", min_value=1, max_value=len(batch_df_raw), value=min(10, len(batch_df_raw)))
                
                disp_df = batch_df_raw.head(top_n).copy()
                
                # Format output for display
                disp_df['Conversion_Probability'] = disp_df['Conversion_Probability'].map(lambda x: f"{x:.1%}")
                
                st.markdown(f"**Showing Top {top_n} Leads:**")
                st.dataframe(disp_df, use_container_width=True)
                
                # Download button for ALL scored leads
                scored_csv = batch_df_raw.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download ALL Scored Leads",
                    data=scored_csv,
                    file_name="scored_leads_output.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")

# --- PAGE 3: XAI EXPLAINABILITY ---
elif navigation == "🧬 XAI Explainability":
    st.markdown("<h1 class='main-title'>🧬 Explainable AI (XAI) Hub</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Explore global feature importance, beeswarm summary plots, and learn the mechanics of explainable AI</p>", unsafe_allow_html=True)
    
    col_xai_left, col_xai_right = st.columns([1, 1])
    
    with col_xai_left:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-bottom: 1rem;">📊 SHAP Global Feature Importance</h3>
                <p style="color: var(--text-color); opacity: 0.9; line-height: 1.6; font-size: 0.95rem;">
                    The bar plot below illustrates the aggregate influence of all features across the entire historical lead database of 5,000 cases.
                    The longer the bar, the more influence that feature has on pushing predictions away from the baseline model average.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Load SHAP global importance plot
        shap_bar_path = os.path.join('reports', 'images', 'shap_bar_importance.png')
        if os.path.exists(shap_bar_path):
            st.image(shap_bar_path, use_container_width=True, caption="Global average absolute impact of features (SHAP Bar Plot)")
        else:
            st.warning("SHAP global plots are being generated. Refresh the page shortly or run `py src/explainable_ai.py` to pre-generate.")
            
    with col_xai_right:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-bottom: 1rem;">🐝 SHAP Summary Plot (Beeswarm)</h3>
                <p style="color: var(--text-color); opacity: 0.9; line-height: 1.6; font-size: 0.95rem;">
                    The beeswarm plot details both feature magnitude AND direction of impact. 
                    Red dots represent high values of that feature, and blue dots represent low values. 
                    If red dots are on the right side, it means high values of that feature increase conversion probability.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        shap_summary_path = os.path.join('reports', 'images', 'shap_summary.png')
        if os.path.exists(shap_summary_path):
            st.image(shap_summary_path, use_container_width=True, caption="SHAP Summary Beeswarm Plot showing direction of impact")
        else:
            st.warning("SHAP summary plot not found.")
            
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-bottom: 1.2rem;">🔬 Deep Explanation of XAI for Non-Technical Stakeholders</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                <div>
                    <h5 style="color: var(--primary-color); font-weight: 600; margin-bottom: 0.5rem;">🔍 What is SHAP?</h5>
                    <p style="color: var(--text-color); opacity: 0.9; font-size: 0.92rem; line-height: 1.5;">
                        <b>SHAP (SHapley Additive exPlanations)</b> is grounded in cooperative game theory. 
                        It treats each lead feature (e.g., response time, budget) as a "player" in a game, 
                        where the "payout" is the predicted probability. SHAP mathematically allocates credit to each player 
                        by measuring how much the prediction changes when that feature is included versus excluded across all possible feature combinations. 
                        This guarantees fair, consistent, and stable attribution.
                    </p>
                </div>
                <div>
                    <h5 style="color: var(--primary-color); font-weight: 600; margin-bottom: 0.5rem;">🍋 What is LIME?</h5>
                    <p style="color: var(--text-color); opacity: 0.9; font-size: 0.92rem; line-height: 1.5;">
                        <b>LIME (Local Interpretable Model-agnostic Explanations)</b> works on a local scale. 
                        If the global ML model is a highly complex, curved wall (non-linear boundary), LIME zooms in extremely close 
                        to a single specific lead. It slightly perturbs (modifies) that lead's attributes and tests the predictions. 
                        By fitting a simple straight-line model (linear surrogate) just for this tiny local neighborhood, LIME shows which features 
                        immediately pull the lead towards conversion or away from it.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- PAGE 4: MODEL COMPARE & ANALYTICS ---
elif navigation == "📊 Model Compare & Analytics":
    st.markdown("<h1 class='main-title'>📊 Model Performance & Historical Lead Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>Compare Logistic Regression vs Random Forest performance and review exploratory insights</p>", unsafe_allow_html=True)
    
    col_mc1, col_mc2 = st.columns([1, 1])
    
    with col_mc1:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-bottom: 1rem;">🏆 Model Metrics Comparison</h3>
                <p style="color: var(--text-color); opacity: 0.9; font-size: 0.95rem; margin-bottom: 1rem;">
                    Both models are evaluated on a holdout test set (20% of dataset). The active model is automatically selected based on superior validation scores.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        comp_img_path = os.path.join('reports', 'images', 'model_comparison.png')
        if os.path.exists(comp_img_path):
            st.image(comp_img_path, use_container_width=True, caption="Model Performance Grouped Bar Chart")
        else:
            st.warning("Comparison plot not pre-rendered.")
            
    with col_mc2:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-bottom: 1rem;">🧩 Confusion Matrices Comparison</h3>
                <p style="color: var(--text-color); opacity: 0.9; font-size: 0.95rem; margin-bottom: 1rem;">
                    Confusion matrices illustrate exactly where models make correct predictions (True Positives, True Negatives) and errors (False Positives, False Negatives).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        cm_img_path = os.path.join('reports', 'images', 'confusion_matrices.png')
        if os.path.exists(cm_img_path):
            st.image(cm_img_path, use_container_width=True, caption="Confusion Matrices Heatmaps")
        else:
            st.warning("Confusion matrix plot not pre-rendered.")
            
    # Business Insights section based on Historical EDA
    st.markdown("<h3 style='color: var(--primary-color); font-family: Outfit; font-weight: 600; margin-top: 2rem; margin-bottom: 1.25rem;'>📈 Exploratory Data Analytics (EDA) Highlights</h3>", unsafe_allow_html=True)
    
    col_eda1, col_eda2, col_eda3 = st.columns(3)
    
    with col_eda1:
        # Conversion rate by Lead Source
        source_conv = df_engineered.groupby('Lead_Source')['Conversion_Status'].mean().reset_index()
        fig_src = px.bar(source_conv, x='Lead_Source', y='Conversion_Status', 
                         title='Lead Conversion Rate by Source',
                         labels={'Conversion_Status':'Conversion Rate'},
                         color_discrete_sequence=['#5A9'])
        fig_src.update_layout()
        st.plotly_chart(fig_src, use_container_width=True)
        
    with col_eda2:
        # Distribution of engagement scores by conversion
        fig_eng = px.box(df_engineered, x='Conversion_Status', y='Engagement_Score',
                         title='Engagement Score Distribution',
                         labels={'Conversion_Status':'Converted (0 = No, 1 = Yes)'},
                         color_discrete_sequence=['#3F72AF'])
        fig_eng.update_layout()
        st.plotly_chart(fig_eng, use_container_width=True)
        
    with col_eda3:
        # Response time vs conversion
        fig_resp = px.histogram(df_engineered, x='Response_Time', color='Conversion_Status',
                                title='Lead Response Time Density',
                                barmode='overlay',
                                labels={'Response_Time':'Response Time (hours)', 'count':'Leads Count'},
                                color_discrete_map={0:'#f85149', 1:'#3fb950'})
        fig_resp.update_layout()
        st.plotly_chart(fig_resp, use_container_width=True)
