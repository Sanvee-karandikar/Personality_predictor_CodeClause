from utils import extract_text_from_pdf
import joblib

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Extract text
cv_path = "cv_input_samples/sanvee_resume.pdf"
text = extract_text_from_pdf(cv_path)

# Predict
features = vectorizer.transform([text])
prediction = model.predict(features)

print("Predicted Personality Type:", prediction[0])
