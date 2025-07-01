import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

st.title("📱 SMS Spam Detection")

# Initialize session variables
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# SMS form
with st.form("sms_form"):
    user_input = st.text_input("Enter an SMS message:", value=st.session_state.user_input, key="input_field")
    submit_btn = st.form_submit_button("Detect Spam")

    if submit_btn and user_input:
        input_vec = vectorizer.transform([user_input.lower()])
        prediction = model.predict(input_vec)[0]
        st.session_state.prediction = prediction
        st.session_state.user_input = ""  # Clear input after submission

# Display prediction
if st.session_state.prediction:
    st.write(f"🧠 Prediction: **{st.session_state.prediction}**")

    if st.button("🔁 Test another message"):
        st.session_state.prediction = None
        st.session_state.user_input = ""
        st.rerun()