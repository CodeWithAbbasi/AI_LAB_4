import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title='Churn Predictor Pro', page_icon='📊', layout='wide')

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        background-color: #3176b1;
        color: white;
        border: none;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #e1eaf2;
        border-radius: 5px 5px 0px 0px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3176b1 !important;
        color: white !important;
    }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Function: Load Model ---
@st.cache_resource
def load_model():
    try:
        with open('best_churn_model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

model = load_model()

EXPECTED_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 
    'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
    'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
    'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
    'TechSupport_No internet service', 'TechSupport_Yes', 
    'StreamingTV_No internet service', 'StreamingTV_Yes', 
    'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check'
]

# --- Header Section ---
st.title('📉 Customer Churn Prediction Dashboard')
st.markdown("##### Strategic Analytics for Customer Retention")
st.divider()

# --- 1. Input Section ---
st.subheader("1. Configure Customer Profile")
tab1, tab2, tab3 = st.tabs(["👤 Personal Info", "📑 Subscription Details", "🌐 Services & Support"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    gender = col1.selectbox('Gender', ['Female', 'Male'])
    senior = col2.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = col3.selectbox('Partner', ['No', 'Yes'])
    dependents = col4.selectbox('Dependents', ['No', 'Yes'])

with tab2:
    c1, c2, c3 = st.columns([2, 1, 1])
    tenure = c1.slider('Tenure (months)', 1, 72, 30)
    contract = c2.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    paperless = c3.selectbox('Paperless Billing', ['No', 'Yes'])
    
    c4, c5 = st.columns(2)
    payment = c4.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = c5.number_input('Monthly Charges ($)', 0.0, 150.0, 65.0)
    total_charges = tenure * monthly_charges

with tab3:
    s_col1, s_col2, s_col3 = st.columns(3)
    internet = s_col1.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    security = s_col1.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    phone = s_col2.selectbox('Phone Service', ['No', 'Yes'])
    backup = s_col2.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    support = s_col3.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    tv = s_col3.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    movies = 'No'; protection = 'No'; lines = 'No'

st.markdown("<br>", unsafe_allow_html=True)

# --- Prediction Logic ---
if st.button('🔍 RUN CHURN ANALYSIS', use_container_width=True):
    if model is None:
        st.error("Model file 'best_churn_model.pkl' not found. Please upload the model to the directory.")
    else:
        input_dict = {
            'SeniorCitizen': 1 if senior == 'Yes' else 0,
            'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
            'gender': gender, 'Partner': partner, 'Dependents': dependents,
            'PhoneService': phone, 'MultipleLines': lines, 'InternetService': internet,
            'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection,
            'TechSupport': support, 'StreamingTV': tv, 'StreamingMovies': movies,
            'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        input_final = input_encoded.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

        prob = model.predict_proba(input_final)[0][1] * 100
        prediction = model.predict(input_final)[0]

        st.divider()
        st.subheader("2. Risk Assessment Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prob < 40:
                risk_level = "LOW RISK"
                risk_color = "green"
            elif prob < 70:
                risk_level = "MEDIUM RISK"
                risk_color = "orange"
            else:
                risk_level = "HIGH RISK"
                risk_color = "red"

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={'font': {'size': 80, 'color': '#7f8c8d'}, 'valueformat': '.1f'},
                # Adjusted Y-domain to start higher (0.18) so number doesn't touch text
                domain={'x': [0, 1], 'y': [0.18, 1]}, 
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                    'bar': {'color': "black", 'thickness': 0.25},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                }
            ))
            
            fig_gauge.update_layout(
                height=400, # Increased height to prevent clipping
                margin=dict(l=50, r=50, t=20, b=20),
                annotations=[dict(
                    text=f"<b>{risk_level}</b>",
                    x=0.5, 
                    y=0.1, # Moved text further down to create space
                    font_size=28, 
                    font_color=risk_color, 
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f"<center>Probability: {prob:.1f}% ({risk_level})</center>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("### Risk Factor Impact")
            impact_data = {
                'Feature': ['Tenure', 'Contract Type', 'Paperless Billing', 'Services & Support'],
                'Impact': [48, 22, 65, 80]
            }
            impact_df = pd.DataFrame(impact_data)
            
            fig_bar = go.Figure(go.Bar(
                x=impact_df['Feature'],
                y=impact_df['Impact'],
                marker_color=['#3176b1', 'green', 'orange', 'red']
            ))
            fig_bar.update_layout(height=350, margin=dict(t=20, b=20), yaxis_title="Risk Weight")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()
        if prediction == 1:
            st.error(f"**Action Recommended:** High churn probability detected. Consider retention offers.")
        else:
            st.success(f"**Retention Outlook:** This customer is stable. Continue standard engagement protocols.")
