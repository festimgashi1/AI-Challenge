import joblib
import pandas as pd

print("=== Text Classification Predictor ===")
print("Loading model...")

try:
    # Load the trained model and vectorizer
    model = joblib.load('models/text_classifier.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    print("Model loaded successfully!")
except:
    print("Error: Model files not found. Please run train_model.py first.")
    exit()

def predict_text(text):
    # Transform the input text
    text_tfidf = tfidf.transform([text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    return prediction, probability

# Interactive prediction
print("\n🔍 Type any message to classify it as SPAM or HAM")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Enter text: ")
    
    if user_input.lower() == 'quit':
        break
        
    prediction, probabilities = predict_text(user_input)
    
    print(f"✅ Prediction: {prediction.upper()}")
    print(f"📊 Confidence - Ham: {probabilities[0]:.2%}, Spam: {probabilities[1]:.2%}")
    print("-" * 50)