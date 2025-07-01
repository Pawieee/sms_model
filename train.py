import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


# Load and clean the dataset
df = pd.read_csv('dataset/spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Simple normalization
df['text'] = df['text'].str.lower()
df.info()
df.label.value_counts() 



# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Convert text to numeric features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

print("Training Success!")

try:
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    joblib.dump(vectorizer, "models/vectorizer.joblib")
    print("Model successfuly dumped!")
except Exception as e:
    print("Error occured: ", e)

