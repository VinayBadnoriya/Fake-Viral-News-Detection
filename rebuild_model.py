import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

print("Loading Fake and Real News dataset (Clément Besnier)...")
df = pd.read_csv("news.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

df.dropna(subset=["text"], inplace=True)
print(f"Shape after dropping nulls: {df.shape}")

print("Label distribution:")
print(df["label"].value_counts())

x = df["text"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

print("Training TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
tf_train = vectorizer.fit_transform(x_train)
tf_test = vectorizer.transform(x_test)

print("Training Passive Aggressive Classifier...")
pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
pac.fit(tf_train, y_train)

y_pred = pac.predict(tf_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Confusion Matrix:")
print(f"              Predicted FAKE   Predicted REAL")
print(f"Actual FAKE   {cm[0, 0]:>6d}        {cm[0, 1]:>6d}")
print(f"Actual REAL   {cm[1, 0]:>6d}        {cm[1, 1]:>6d}")
print(f"\nFalse positives (REAL predicted as FAKE): {cm[1, 0]}")
print(f"False negatives (FAKE predicted as REAL): {cm[0, 1]}")

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(pac, "fake_news_model.pkl")
joblib.dump(pac, "finalized_model.pkl")

print("\nModel and vectorizer saved successfully!")
