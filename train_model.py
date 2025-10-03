import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

print("=== AI Internship Challenge - Text Classification ===")

print("1. Loading dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=['label', 'text'])

print(f"Dataset loaded: {len(df)} samples")
print("Category distribution:")
print(df['label'].value_counts())

print("\n2. Exploratory Data Analysis...")

from collections import Counter
import re

def get_top_words(texts, n=10):
    words = []
    for text in texts:
        words.extend(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
    return Counter(words).most_common(n)

print("\nTop 10 words in spam messages:")
spam_texts = df[df['label'] == 'spam']['text']
spam_words = get_top_words(spam_texts)
for word, count in spam_words[:10]:
    print(f"  {word}: {count}")

print("\nTop 10 words in ham messages:")
ham_texts = df[df['label'] == 'ham']['text']
ham_words = get_top_words(ham_texts)
for word, count in ham_words[:10]:
    print(f"  {word}: {count}")

print("\n3. Preprocessing text and extracting features...")
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    lowercase=True
)

X = tfidf.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("\n4. Training models...")

print("   - Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)

print("   - Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n5. Model Comparison:")
print(f"Naive Bayes Accuracy: {nb_accuracy:.2%}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.2%}")

if lr_accuracy > nb_accuracy:
    best_model = lr_model
    best_model_name = "Logistic Regression"
    best_predictions = lr_pred
else:
    best_model = nb_model
    best_model_name = "Naive Bayes"
    best_predictions = nb_pred

print(f"\nBest model: {best_model_name}")

print("\n6. Detailed Evaluation:")
print(classification_report(y_test, best_predictions))

print("\n7. Creating confusion matrix...")
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['ham', 'spam'], 
            yticklabels=['ham', 'spam'])
plt.title('Confusion Matrix - Spam Detection')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

print("Confusion matrix saved as 'confusion_matrix.png'")

print("\n8. Saving model...")
os.makedirs('models', exist_ok=True)

joblib.dump(best_model, 'models/text_classifier.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

print("Model saved to: models/text_classifier.pkl")
print("Vectorizer saved to: models/tfidf_vectorizer.pkl")

print("\n🎉 Training completed successfully!")
print(f"📊 Best model accuracy: {max(nb_accuracy, lr_accuracy):.2%}")
