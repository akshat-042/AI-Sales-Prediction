"""
Data Generator for Sales Conversion Prediction Project.
Author: Antigravity AI Coding Assistant
Description: This script generates a highly realistic sales lead dataset with
             built-in statistical relationships. It represents various business
             factors that influence lead conversion.
"""

import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_leads_dataset(num_samples=5000):
    """
    Generates a realistic synthetic sales leads dataset with built-in correlations.
    
    Parameters:
    -----------
    num_samples : int
        Number of leads to generate (default: 5000)
        
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the synthesized lead data.
    """
    print(f"Generating {num_samples} realistic sales leads...")
    
    # 1. Generate core columns with realistic distributions
    lead_ids = [f"LEAD{i:04d}" for i in range(1, num_samples + 1)]
    
    # Categorical distributions with probabilities
    lead_sources = np.random.choice(
        ['Website', 'Email', 'Cold Call', 'LinkedIn', 'Referral'],
        size=num_samples,
        p=[0.35, 0.25, 0.15, 0.18, 0.07]
    )
    
    industries = np.random.choice(
        ['Tech', 'Finance', 'Healthcare', 'Retail', 'Education', 'Manufacturing'],
        size=num_samples,
        p=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
    )
    
    company_sizes = np.random.choice(
        ['Small', 'Medium', 'Large'],
        size=num_samples,
        p=[0.50, 0.35, 0.15]
    )
    
    locations = np.random.choice(
        ['North America', 'Europe', 'Asia-Pacific', 'South America', 'Middle East'],
        size=num_samples,
        p=[0.40, 0.25, 0.20, 0.08, 0.07]
    )
    
    # Continuous numeric distributions
    # Budget: log-normal-like distribution, mostly small-medium budgets, occasionally high
    budgets = np.random.exponential(scale=20000, size=num_samples) + 2000
    budgets = np.clip(budgets, 1000, 150000).round(-2) # round to nearest hundred
    
    # Interaction Count: Poisson distribution (average 5 interactions)
    interaction_counts = np.random.poisson(lam=5, size=num_samples) + 1
    interaction_counts = np.clip(interaction_counts, 1, 25)
    
    # Response Time: Exponential distribution (hours)
    response_times = np.random.exponential(scale=12, size=num_samples) + 0.1
    response_times = np.clip(response_times, 0.1, 48.0).round(1)
    
    # Engagement Score: Uniform-like distribution with standard deviation
    engagement_scores = np.random.normal(loc=55, scale=20, size=num_samples)
    engagement_scores = np.clip(engagement_scores, 1, 100).round().astype(int)
    
    # Previous Contact History: Number of interactions in past marketing campaigns
    prev_contacts = np.random.binomial(n=5, p=0.15, size=num_samples)
    
    # 2. Build the target variable 'Conversion_Status' using statistical relationships
    # We define a logit function to calculate conversion probability for each record.
    # This ensures that ML models can discover clean, explainable relationships.
    
    probabilities = []
    
    for i in range(num_samples):
        # Base logit score (log-odds) equivalent to a baseline probability of ~20%
        # logit(0.20) = ln(0.20 / 0.80) = -1.38
        score = -1.38
        
        # Lead Source effect
        source = lead_sources[i]
        if source == 'Referral':
            score += 1.50   # Referral is highly likely to convert
        elif source == 'Website':
            score += 0.60   # Inbound web inquiries are warm
        elif source == 'LinkedIn':
            score += 0.40   # Professional network outreach
        elif source == 'Email':
            score += 0.00   # Email campaigns are neutral
        elif source == 'Cold Call':
            score -= 0.80   # Outbound cold calls have low conversion
            
        # Industry effect
        ind = industries[i]
        if ind == 'Tech':
            score += 0.30
        elif ind == 'Finance':
            score += 0.20
        elif ind == 'Healthcare':
            score += 0.15
        elif ind == 'Education':
            score -= 0.25
            
        # Company Size effect
        c_size = company_sizes[i]
        if c_size == 'Large':
            score += 0.40
        elif c_size == 'Medium':
            score += 0.15
        else:
            score -= 0.20
            
        # Budget effect (log scale helps representing diminishing returns of high budgets)
        budget = budgets[i]
        score += np.log10(budget / 1000.0) * 0.40
        
        # Interaction Count effect
        interactions = interaction_counts[i]
        if interactions >= 8:
            score += 0.80
        elif interactions >= 4:
            score += 0.40
        else:
            score -= 0.60
            
        # Response Time effect
        resp_time = response_times[i]
        if resp_time <= 1.0:
            score += 1.00   # Immediate response increases odds dramatically
        elif resp_time <= 4.0:
            score += 0.45   # Quick response is good
        elif resp_time <= 24.0:
            score -= 0.20   # Delayed response hurts odds
        else:
            score -= 0.90   # Neglected lead
            
        # Engagement Score effect
        eng_score = engagement_scores[i]
        score += (eng_score - 50) / 25.0  # Linear influence relative to mean
        
        # Previous Contact History effect
        prev_cnt = prev_contacts[i]
        if 1 <= prev_cnt <= 3:
            score += 0.35   # Nurtured leads convert better
        elif prev_cnt > 3:
            score -= 0.15   # Alert lead fatigue (too many contacts)
            
        # Add random normal noise to make predictions non-deterministic
        score += np.random.normal(loc=0.0, scale=0.45)
        
        # Sigmoid function to convert log-odds to probability
        prob = 1.0 / (1.0 + np.exp(-score))
        probabilities.append(prob)
        
    probabilities = np.array(probabilities)
    
    # Generate binary conversion status using the calculated probabilities
    conversions = np.random.binomial(n=1, p=probabilities)
    
    # 3. Assemble and return the DataFrame
    df = pd.DataFrame({
        'Lead_ID': lead_ids,
        'Lead_Source': lead_sources,
        'Industry': industries,
        'Company_Size': company_sizes,
        'Budget': budgets,
        'Interaction_Count': interaction_counts,
        'Response_Time': response_times,
        'Location': locations,
        'Engagement_Score': engagement_scores,
        'Previous_Contact_History': prev_contacts,
        'Conversion_Probability_True': probabilities.round(4),  # Saved for internal validation
        'Conversion_Status': conversions
    })
    
    return df

if __name__ == '__main__':
    # Create direct paths
    raw_data_dir = os.path.join('data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Generate and save
    df_leads = generate_leads_dataset(num_samples=5000)
    output_path = os.path.join(raw_data_dir, 'sales_leads_raw.csv')
    df_leads.to_csv(output_path, index=False)
    print(f"Dataset generated successfully and saved to: {output_path}")
    print("\nDataset Summary Statistics:")
    print(df_leads.describe())
    print("\nConversion Rate:")
    print(df_leads['Conversion_Status'].value_counts(normalize=True))
