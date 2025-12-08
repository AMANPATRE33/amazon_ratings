import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- 1. Configuration and Setup (CRITICAL: Must Match Training) ---

# Use the fixed filename
MODEL_FILENAME = "amazon_quality_classifier_final.pkl" 

# 🛑 FINAL CORRECT FEATURE LIST: All 15+ columns in the correct order.
FEATURE_COLUMNS = [
    'Discounted Price', 
    'Discount Percentage', 
    'Rating Count', 
    'Cat_Car&Motorbike', 
    'Cat_Computers&Accessories', 
    'Cat_Electronics', 
    'Cat_Health&PersonalCare', 
    'Cat_Home&Kitchen', 
    'Cat_HomeImprovement', 
    'Cat_MusicalInstruments', 
    'Cat_OfficeProducts', 
    'Cat_Toys&Games', 
    'Price_Per_RatingCount', 
    'Log_Dollar_Saved', 
    'Log_Rating_Count'
]

# List of categories for the dropdown menu
CATEGORY_NAMES = [col.replace('Cat_', '') for col in FEATURE_COLUMNS if col.startswith('Cat_')]


# 2. Model Loading Function
@st.cache_resource
def load_model():
    """Loads the model once and caches it."""
    try:
        with open(MODEL_FILENAME, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()


# 3. Prediction Logic
def make_prediction(input_data):
    """Recreates all engineered features and makes a prediction."""
    if model is None:
        return 0

    features = {col: 0 for col in FEATURE_COLUMNS}
    
    # Extract raw inputs
    dp = input_data['discounted_price']
    ap = input_data['actual_price']
    rc = input_data['rating_count']
    dp_perc = input_data['discount_percentage']
    
    # 3a. Populate Core Features
    features['Discounted Price'] = dp
    features['Rating Count'] = rc
    features['Discount Percentage'] = dp_perc

    # 3b. Recreate Advanced Engineered Features
    features['Price_Per_RatingCount'] = dp / (rc + 1)
    dollar_saved = ap - dp
    features['Log_Dollar_Saved'] = np.log1p(dollar_saved)
    features['Log_Rating_Count'] = np.log1p(rc)

    # 3c. Populate One-Hot Encoded Feature
    selected_category_col = f'Cat_{input_data["category"]}'
    if selected_category_col in features:
        features[selected_category_col] = 1
    
    # Convert to DataFrame using the EXACT required column list
    input_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    return prediction, prediction_proba[1] # Return class and high-quality probability


# 4. Streamlit UI Layout
st.set_page_config(page_title="Amazon Quality Predictor", layout="wide")

st.title("🛒 Amazon Product Quality Predictor")
st.markdown("---")
st.subheader("Predicting High Quality (Rating > 4.0) with Advanced Features")


if model:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Price and Popularity Metrics")
        discounted_price = st.number_input("1. Discounted Price ($)", min_value=0.0, value=85.00, step=0.01)
        actual_price = st.number_input("2. Actual Price ($)", min_value=0.0, value=100.00, step=0.01)
        rating_count = st.number_input("3. Rating Count (Volume)", min_value=0, value=12500)
        
    with col2:
        st.markdown("#### Discount and Category Details")
        discount_percentage = st.number_input("4. Discount Percentage (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
        
        st.write("") 
        category = st.selectbox("5. Product Category", CATEGORY_NAMES)
        st.markdown("<br>", unsafe_allow_html=True)


    if st.button("ANALYZE PRODUCT VALUE", use_container_width=True):
        
        input_data = {
            'discounted_price': discounted_price,
            'actual_price': actual_price,
            'rating_count': rating_count,
            'discount_percentage': discount_percentage,
            'category': category
        }
        
        prediction, confidence = make_prediction(input_data)
        
        st.markdown("---")
        
        if prediction == 1:
            st.success(f"✅ Prediction: HIGH QUALITY (Rating > 4.0)", icon="⭐")
            st.markdown(f"<h3 style='color: green; text-align: center;'>Confidence: {confidence:.2f}%</h3>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.warning(f"⚠️ Prediction: AVERAGE/POOR QUALITY (Rating ≤ 4.0)", icon="👎")
            st.markdown(f"<h3 style='color: #FF4B4B; text-align: center;'>Confidence (High Quality): {confidence:.2f}%</h3>", unsafe_allow_html=True)