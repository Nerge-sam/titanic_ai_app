import streamlit as st
import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import requests
import joblib # <--- NEW: To load the brain
from datetime import datetime
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Titanic AI Explorer", layout='wide', page_icon="🚢")
load_dotenv()

# --- 2. BACKEND (Data & Model) ---
@st.cache_data
def load_data():
    """Loads the Titanic dataset."""
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    df = pd.read_csv(url)
    return df

@st.cache_resource
def load_model():
    """Loads the Machine Learning Model (Brain)."""
    try:
        model = joblib.load('titanic_model.pkl')
        return model
    except Exception as e:
        return None

df = load_data()
model = load_model() # <--- Load the brain immediately

# --- 3. AI FUNCTIONS (Text Generation) ---
def analyze_sentiment(text):
    """Analyzes text using the robust Fallback strategy."""
    if not text or not text.strip(): return "Neutral 😐"
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    api_url = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": text}, timeout=3)
        if response.status_code == 200:
            label_map = {"LABEL_0": "Negative 😡", "LABEL_1": "Neutral 😐", "LABEL_2": "Positive 😃"}
            best_label = response.json()[0][0]['label']
            return label_map.get(best_label, "Neutral 😐")
    except: pass
    if any(w in text.lower() for w in ['bad', 'slow', 'error']): return "Negative 😡"
    if any(w in text.lower() for w in ['good', 'great', 'love']): return "Positive 😃"
    return "Neutral 😐"

def generate_ai_report(total, survivors, rate, sex_filter):
    """Generates a text report."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    try:
        api_url = "https://router.huggingface.co/hf-inference/models/HuggingFaceH4/zephyr-7b-beta"
        headers = {"Authorization": f"Bearer {token}"}
        prompt = f"<|system|>Data Analyst</s><|user|>Analyze: Group {sex_filter}, Rate {rate}%. One sentence summary.</s><|assistant|>"
        response = requests.post(api_url, headers=headers, json={"inputs": prompt}, timeout=4)
        if response.status_code == 200: return response.json()[0]['generated_text'].strip()
    except: pass
    return f"ℹ️ The {sex_filter} group had a survival rate of {rate}%."

# --- 4. MAIN APP UI ---
try:
    st.title("🚢 Titanic Survival Explorer")

    # --- GLOBAL FILTERS ---
    st.subheader("⚙️ Filter Options (For Dashboard)")
    f1, f2, f3, f4 = st.columns(4)
    with f1: selected_sex = st.selectbox("Select Gender:", ['male', 'female'], key='sex_filter')
    with f2: selected_age = st.slider("Max Age:", 1, 80, 35, key='age_filter')
    with f3: selected_class = st.selectbox("Class", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x==1 else f"{x}nd" if x==2 else "3rd", key='class_filter')
    with f4: selected_fare = st.slider("Max Fare:", 0, 150, 85, key='fare_filter')

    filtered_df = df[(df['sex'] == selected_sex) & (df['age'] <= selected_age) & (df['pclass'] == selected_class) & (df['fare'] <= selected_fare)]

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📝 AI Report", "🔮 Predict Survival", "🤖 Feedback"])

    # === TAB 1: DASHBOARD ===
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader(f"📋 Passenger List ({len(filtered_df)})")
            st.dataframe(filtered_df[['survived', 'pclass', 'sex', 'age', 'fare']].head(10), hide_index=True)
        with col2:
            st.subheader("📈 Survival Chart")
            fig, ax = plt.subplots()
            if not filtered_df.empty:
                chart_df = filtered_df['survived'].map({0: "Did Not Survive", 1: "Survived"})
                status_counts = chart_df.value_counts()
                colors = [{"Survived": "#4ecdc4", "Did Not Survive": "#ff6b6b"}.get(x, "#999") for x in status_counts.index]
                status_counts.plot.pie(autopct='%1.1f%%', colors=colors, ax=ax)
                ax.set_ylabel('')
                st.pyplot(fig)
            else: st.warning("No data found.")
        with col3:
            st.subheader("📊 Key Metrics")
            total = len(filtered_df)
            survived = filtered_df['survived'].sum()
            rate = (survived / total * 100) if total > 0 else 0
            st.metric("Total Passengers", total)
            st.metric("Survivors", survived)
            st.metric("Survival Rate", f"{rate:.1f}%")

    # === TAB 2: REPORT ===
    with tab2:
        st.header("🤖 AI Analysis")
        if st.button("Generate Report"):
            with st.spinner("Analyzing..."):
                total = len(filtered_df)
                survived = filtered_df['survived'].sum()
                rate = (survived / total * 100) if total > 0 else 0
                report = generate_ai_report(total, survived, f"{rate:.1f}", selected_sex)
                st.info(report)

    # === TAB 3: PREDICTION (NEW!) ===
    with tab4: # Using tab4 variable name for clarity, but UI label is "Feedback"
        pass 
    
    with tab3:
        st.header("🔮 Would YOU Survive?")
        st.write("Enter your details to ask the Machine Learning Model.")
        
        # 1. Inputs specifically for the model
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1: p_class = st.selectbox("Ticket Class", [1, 2, 3], format_func=lambda x: f"{x}st Class")
        with p_col2: p_sex = st.selectbox("Gender", ["Male", "Female"])
        with p_col3: p_age = st.slider("Your Age", 1, 100, 25)
        with p_col4: p_fare = st.slider("Ticket Price ($)", 0, 500, 50)

        # 2. The Predict Button
        if st.button("🔮 Predict My Fate"):
            if model is not None:
                # Prepare data exactly how the model was trained
                # Model expects: ['pclass', 'sex', 'age', 'fare']
                # Sex must be: male=0, female=1
                sex_val = 1 if p_sex == "Female" else 0
                
                input_data = pd.DataFrame([[p_class, sex_val, p_age, p_fare]], 
                                        columns=['pclass', 'sex', 'age', 'fare'])
                
                # Predict
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1] # Probability of surviving
                
                # Show Result
                st.markdown("---")
                if prediction == 1:
                    st.success(f"🎉 **YOU SURVIVED!** (Survival Probability: {probability:.1%})")
                    st.balloons()
                else:
                    st.error(f"💀 **You did not survive.** (Survival Probability: {probability:.1%})")
            else:
                st.error("⚠️ Model not found! Please run train_model.py first.")

    # === TAB 4: FEEDBACK ===
    with tab4:
        st.header("💬 Feedback")
        user_text = st.text_area("Experience?", placeholder="Type here...")
        if st.button("Analyze Sentiment"):
            st.info(analyze_sentiment(user_text))

except Exception as e:
    st.error("⚠️ An unexpected error occurred! Please refresh.")
