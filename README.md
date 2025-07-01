# 📩 SMS Spam Detection with Multinomial Naive Bayes


![Image Preview](images\preview.png)
This is a simple machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using the **Multinomial Naive Bayes** algorithm. The app is built with **Streamlit** for an easy-to-use web interface.

---

## 🚀 Demo

Live App: _[https://sms-model-jack-n-dio.streamlit.app](#)_


---

## 🧠 How It Works

1. The model is trained using a labeled dataset of SMS messages.
2. Messages are preprocessed (cleaned, tokenized, vectorized).
3. The MultinomialNB model is used to predict whether a message is spam or not.
4. Users can input their own SMS message in the app to see predictions in real time.

---

## 🛠 Features

- ✔️ Real-time SMS classification
- ✔️ Clean and minimal interface with Streamlit
- ✔️ Uses **CountVectorizer** for feature extraction
- ✔️ Lightweight and fast with **Multinomial Naive Bayes**


---

## ⚙️ Run Locally

1. Clone the repository
    ```bash
    git clone https://github.com/your-username/sms_model.git
    cd sms_model
    ```

2. Install dependencies
    ```bash
    pip install -r "requirements.txt"
    ```

3. Run streamlit
    ```bash
    streamlit run app.py
    ```

## Requirements

    - Python installed locally
    - Pandas
    - Scikit-learn
    - Streamlit
    - Joblib
