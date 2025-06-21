from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# Sample training data (you can expand this later)
texts = [
    "Team player, communicates well, works in groups",
    "Leads teams, makes decisions, mentors others",
    "Independent worker, works without supervision",
    "Organized, follows structure, meets deadlines",
    "Creative thinker, adapts easily, accepts feedback"
]
labels = ["Agreeable", "Extrovert", "Introvert", "Conscientious", "Open"]

# Vectorize the text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train the model
model = MultinomialNB()
model.fit(X, labels)

# ✅ Create 'model' folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the model and vectorizer
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model trained and saved successfully!")
