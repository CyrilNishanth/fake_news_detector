import pandas as pd

# Load both files
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# Add labels
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Combine them
df = pd.concat([fake_df, true_df], ignore_index=True)

# Show basic info
print("📊 Combined dataset shape:", df.shape)
print("\n📝 Columns:", df.columns.tolist())
print("\n📌 First 5 rows:")
print(df.head())
#Pre-Processing of the text
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(preprocess)

print("✅ Cleaned sample:")
print(df[["text", "clean_text"]].head())
#TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine title and text into one column for better representation
df['content'] = df['title'] + " " + df['text']

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the content column
X = tfidf.fit_transform(df['content'])

# Set y as the label column (target)
y = df['label']
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))
# Save the model
import os
import joblib

# Create 'model' folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model and vectorizer
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")

