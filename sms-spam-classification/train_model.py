import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load your dataset
df = pd.read_csv("spam.csv", encoding="latin1")
  # change to your dataset name
df = df.iloc[:, :2]

# Rename columns if needed

df.columns = ["label", "text"]

# Convert labels into numbers (ham=0, spam=1)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save both files
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("DONE — model.pkl and vectorizer.pkl created!")
