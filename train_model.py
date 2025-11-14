import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example training data (you can expand this list later)
training_sentences = [
    "hi", "hello", "hey",
    "how are you", "what's up",
    "what's for dinner", "i am hungry", "let's eat",
    "bye", "goodbye", "see you later"
]

training_labels = [
    "greeting",
    "greeting",
    "greeting",
    "wellbeing",
    "wellbeing",
    "food",
    "food",
    "food",
    "farewell",
    "farewell",
    "farewell"
]

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)
y = training_labels

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Save vectorizer and model
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(model, "naive_bayes_model.joblib")

print("✅ Model and vectorizer saved successfully!")
