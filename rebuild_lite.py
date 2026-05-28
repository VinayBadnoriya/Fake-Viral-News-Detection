import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading dataset...")
df = pd.read_csv("news.csv")
df.dropna(subset=["text"], inplace=True)

x = df["text"]
y = df["label"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")
print("Training lite TF-IDF Vectorizer (max_features=3000)...")

vectorizer = TfidfVectorizer(
    stop_words='english', max_df=0.7, max_features=3000, ngram_range=(1, 2)
)
tf_train = vectorizer.fit_transform(x_train)
tf_test = vectorizer.transform(x_test)

print("Training Passive Aggressive Classifier...")
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
pac.fit(tf_train, y_train)

y_pred = pac.predict(tf_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(pac, "fake_news_model.pkl")

import os
vec_size = os.path.getsize("tfidf_vectorizer.pkl") / (1024 * 1024)
model_size = os.path.getsize("fake_news_model.pkl") / (1024 * 1024)
print(f"Vectorizer: {vec_size:.1f} MB, Model: {model_size:.1f} MB")
print("Lite models saved successfully!")
