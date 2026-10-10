import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ==========================================
# PAGE CONFIGURATION & MODERN CLASSIC THEME
# ==========================================
st.set_page_config(
    page_title='365 Automobiles | ML Intelligence Hub',
    page_icon='🚗',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Custom CSS for Modern Classic Dark Theme & Glassmorphism
st.markdown(
    """
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-weight: 500;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 700;
    }

    /* Headers & Titles */
    h1, h2, h3 {
        color: #e6edf3;
        font-family: 'Playfair Display', serif;
    }
    
    /* Custom Containers / Cards */
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
        box-shadow: 0 0 15px rgba(46, 160, 67, 0.4);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# DATA LOADING & MODEL CACHING
# ==========================================
@st.cache_data
def load_data():
  sales = pd.read_csv('365_Automobiles_Sales.csv')
  service = pd.read_csv('365_Automobiles_Service.csv')
  return sales, service


sales_df, service_df = load_data()


@st.cache_resource
def train_models(sales_df):
  # Model A: Linear Regression (Price)
  buyers = sales_df[sales_df['purchased'] == 'Yes'].copy()
  features_a = [
      'customer_age',
      'monthly_income_aed',
      'years_at_current_job',
      'battery_kwh',
      'range_km',
      'motor_power_hp',
      'list_price_aed',
      'options_value_aed',
      'showroom_visits',
      'test_drive_taken',
      'brochure_downloaded',
      'down_payment_pct',
  ]
  X_a = pd.get_dummies(buyers[features_a], drop_first=True)
  y_a = buyers['final_price_aed']
  lr = LinearRegression()
  lr.fit(X_a, y_a)

  # Model B: Lead Conversion RF
  features_b = [
      'customer_age',
      'monthly_income_aed',
      'showroom_visits',
      'test_drive_taken',
      'brochure_downloaded',
      'finance_quote_requested',
      'days_in_pipeline',
      'has_tradein',
      'salesperson_experience_yrs',
  ]
  X_b = pd.get_dummies(sales_df[features_b], drop_first=True)
  y_b = sales_df['purchased'].apply(lambda x: 1 if x == 'Yes' else 0)
  rf_b = RandomForestClassifier(random_state=42)
  rf_b.fit(X_b, y_b)

  return lr, rf_b, features_a, features_b, X_a.columns


lr_model, rf_lead_model, feat_a, feat_b, x_cols_a = train_models(sales_df)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown(
    "<h2 style='text-align: center; color: #58a6ff;'>🚗 365 AUTOMOBILES</h2>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<p style='text-align: center; color: #8b949e;'>Executive AI Command"
    " Center</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown('---')

menu = st.sidebar.radio(
    'Navigation',
    [
        '📊 Executive Dashboard',
        '💵 Deal Price Estimator (Model A)',
        '🎯 Lead Scorer (Model B)',
        '🛠️ After-Sales & Service Intel',
    ],
)

# ==========================================
# 1. EXECUTIVE DASHBOARD
# ==========================================
if menu == '📊 Executive Dashboard':
  st.title('📊 Executive Business Intelligence Dashboard')
  st.markdown(
      'Real-time commercial analytics across UAE showrooms and service'
      ' centres.'
  )

  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric('Total Enquiries', f'{len(sales_df):,}')
  with col2:
    buyers_count = len(sales_df[sales_df['purchased'] == 'Yes'])
    st.metric('Converted Buyers', f'{buyers_count:,}')
  with col3:
    conv_rate = (buyers_count / len(sales_df)) * 100
    st.metric('Conversion Rate', f'{conv_rate:.1f}%')
  with col4:
    avg_price = sales_df['final_price_aed'].mean()
    st.metric('Avg. Final Deal Price', f'AED {avg_price:,.0f}')

  st.markdown('---')

  col_l, col_r = st.columns(2)

  with col_l:
    st.subheader('Enquiries & Conversions by Showroom')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=sales_df,
        x='showroom_location',
        hue='purchased',
        palette={'Yes': '#2ea043', 'No': '#da3633'},
        ax=ax,
    )
    ax.set_facecolor('#161b22')
    fig.patch.set_facecolor('#161b22')
    ax.tick_params(colors='white', axis='x', rotation=15)
    ax.tick_params(colors='white', axis='y')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.legend(facecolor='#161b22', labelcolor='white')
    st.pyplot(fig)

  with col_r:
    st.subheader('Lead Sources Performance')
    fig, ax = plt.subplots(figsize=(8, 5))
    lead_perf = (
        sales_df.groupby('lead_source')['purchased']
        .apply(lambda x: (x == 'Yes').mean() * 100)
        .reset_index()
    )
    sns.barplot(
        data=lead_perf,
        x='lead_source',
        y='purchased',
        palette='crest',
        ax=ax,
    )
    ax.set_facecolor('#161b22')
    fig.patch.set_facecolor('#161b22')
    ax.set_ylabel('Conversion Rate (%)')
    ax.tick_params(colors='white', axis='x', rotation=15)
    ax.tick_params(colors='white', axis='y')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    st.pyplot(fig)

# ==========================================
# 2. MODEL A: DEAL PRICE ESTIMATOR
# ==========================================
elif menu == '💵 Deal Price Estimator (Model A)':
  st.title('💵 Final Selling Price Estimator (Linear Regression)')
  st.markdown(
      'Predict final negotiated vehicle selling prices prior to agreement'
      '[cite: 4].'
  )

  with st.form('price_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
      list_price = st.number_input(
          'List Price (AED)', 150000, 600000, 280000, step=10000
      )
      options_val = st.number_input(
          'Options Value (AED)', 5000, 50000, 15000, step=1000
      )
      battery = st.number_input('Battery Capacity (kWh)', 60.0, 120.0, 85.0)
    with col2:
      range_km = st.number_input('Range (km)', 300, 700, 480)
      motor_hp = st.number_input('Motor Power (HP)', 250, 750, 400)
      down_payment_pct = st.slider('Down Payment (%)', 10.0, 50.0, 20.0)
    with col3:
      customer_age = st.slider('Customer Age', 21, 70, 35)
      monthly_income = st.number_input(
          'Monthly Income (AED)', 10000, 150000, 35000, step=5000
      )
      years_job = st.slider('Years at Current Job', 0.0, 20.0, 3.0)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
      showroom_visits = st.slider('Showroom Visits', 1, 5, 2)
    with col_b:
      test_drive = st.selectbox('Test Drive Taken?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    with col_c:
      brochure = st.selectbox('Brochure Downloaded?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    submitted = st.form_submit_button('Calculate Predicted Final Price')

    if submitted:
      input_data = pd.DataFrame(
          [[
              customer_age,
              monthly_income,
              years_job,
              battery,
              range_km,
              motor_hp,
              list_price,
              options_val,
              showroom_visits,
              test_drive,
              brochure,
              down_payment_pct,
          ]],
          columns=feat_a,
      )
      input_encoded = pd.get_dummies(input_data).reindex(
          columns=x_cols_a, fill_value=0
      )
      pred_price = lr_model.predict(input_encoded)[0]

      st.success(f'### Predicted Final Deal Price: AED {pred_price:,.2f}')

# ==========================================
# 3. MODEL B: LEAD CONVERSION SCORER
# ==========================================
elif menu == '🎯 Lead Scorer (Model B)':
  st.title('🎯 Lead Conversion Scorer (Random Forest)')
  st.markdown(
      'Evaluate incoming leads in real-time to prioritize high-intent sales'
      ' efforts[cite: 5].'
  )

  with st.form('lead_form'):
    col1, col2 = st.columns(2)
    with col1:
      age_l = st.slider('Lead Age', 20, 65, 38)
      income_l = st.number_input(
          'Lead Monthly Income (AED)', 8000, 120000, 30000, step=5000
      )
      visits_l = st.slider('Showroom Visits (Lead)', 0, 5, 1)
      test_l = st.selectbox('Test Drive Taken (Lead)', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='tl')
    with col2:
      brochure_l = st.selectbox('Brochure Downloaded (Lead)', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='bl')
      quote_l = st.selectbox('Finance Quote Requested', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
      pipeline_l = st.slider('Days in Pipeline', 1, 90, 15)
      tradein_l = st.selectbox('Has Trade-in?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
      sales_exp = st.slider("Salesperson's Experience (Years)", 0.5, 15.0, 4.0)

    lead_submitted = st.form_submit_button('Score Lead Conversion Probability')

    if lead_submitted:
      input_lead = pd.DataFrame(
          [[
              age_l,
              income_l,
              visits_l,
              test_l,
              brochure_l,
              quote_l,
              pipeline_l,
              tradein_l,
              sales_exp,
          ]],
          columns=feat_b,
      )
      input_lead_enc = pd.get_dummies(input_lead)
      # align columns with training
      X_dummy = pd.get_dummies(
          sales_df[feat_b], drop_first=True
      )
      input_lead_enc = input_lead_enc.reindex(
          columns=X_dummy.columns, fill_value=0
      )

      prob = rf_lead_model.predict_proba(input_lead_enc)[0][1]

      st.markdown('---')
      if prob >= 0.65:
        st.success(
            f'### High-Intent Lead! Conversion Probability: {prob * 100:.1f}%'
        )
        st.markdown(
            '**Recommendation:** Assign to senior sales executive immediately and'
            ' offer VIP test drive.'
        )
      elif prob >= 0.35:
        st.warning(
            f'### Moderate Lead. Conversion Probability: {prob * 100:.1f}%'
        )
        st.markdown(
            '**Recommendation:** Nurture via automated email campaigns and'
            ' brochure follow-ups.'
        )
      else:
        st.error(f'### Low-Priority Lead. Conversion Probability: {prob * 100:.1f}%')
        st.markdown(
            '**Recommendation:** Keep in standard pipeline; do not allocate'
            ' expensive showroom resources.'
        )

# ==========================================
# 4. AFTER-SALES & SERVICE INTEL
# ==========================================
elif menu == '🛠️ After-Sales & Service Intel':
  st.title('🛠️ After-Sales & Service Centre Analytics')
  st.markdown(
      'Monitor workshop visits, battery degradation, and customer satisfaction'
      '[cite: 6].'
  )

  col1, col2 = st.columns(2)
  with col1:
    st.subheader('Average Service Cost by Centre')
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=service_df,
        x='service_centre',
        y='service_cost_aed',
        palette='magma',
        ax=ax,
    )
    ax.set_facecolor('#161b22')
    fig.patch.set_facecolor('#161b22')
    ax.tick_params(colors='white', axis='x', rotation=10)
    ax.tick_params(colors='white', axis='y')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    st.pyplot(fig)

  with col2:
    st.subheader('Battery Health % vs. Vehicle Age (Months)')
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=service_df,
        x='vehicle_age_months',
        y='battery_health_pct',
        alpha=0.5,
        color='#58a6ff',
        ax=ax,
    )
    ax.set_facecolor('#161b22')
    fig.patch.set_facecolor('#161b22')
    ax.tick_params(colors='white', axis='x')
    ax.tick_params(colors='white', axis='y')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    st.pyplot(fig)