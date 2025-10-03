# AI Challenge - Text Classification

A machine learning system for spam detection that classifies text messages as "spam" or "ham" (not spam). Built for the Linkplus IT AI Internship Challenge.

## 🚀 Features

- **Data Preprocessing**: Text cleaning and normalization
- **Feature Extraction**: TF-IDF vectorization for text representation
- **Model Training**: Multiple algorithm comparison (Naive Bayes vs Logistic Regression)
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score
- **Visualization**: Confusion matrix for performance analysis
- **Interactive Prediction**: Real-time text classification with confidence scores
- **Model Persistence**: Save and load trained models for deployment

## 📊 Results

- **Best Model Accuracy**: 97.76% (Naive Bayes)
- **Spam Detection Recall**: 86%
- **Ham Detection Recall**: 100%

### Classification Report:
text
          precision    recall  f1-score   support

     ham       0.98      1.00      0.99       966
    spam       0.97      0.86      0.91       149

accuracy                           0.98      1115
macro avg 0.97 0.93 0.95 1115
weighted avg 0.98 0.98 0.98 1115

text

## 🛠️ Installation & Setup

### Step 1: Install Python
Download and install Python from [python.org](https://python.org). During installation, check "Add Python to PATH".

### Step 2: Install Required Libraries
Open Command Prompt and run:
```bash
pip install pandas scikit-learn matplotlib seaborn joblib
Step 3: Download the Project
Download or clone this project to your computer.

🎯 How to Run the Program
Option 1: Complete Training & Prediction
First, train the model:

bash
python train_model.py
This will download the dataset, train the AI model, and save it (takes 1-2 minutes).

Then, make predictions:

bash
python predict.py
Type any message and press Enter to see if it's SPAM or HAM.

Option 2: Use Pre-trained Model
If the model is already trained, just run:

bash
python predict.py
📁 Project Structure
text
ai-test/
├── train_model.py          # Main training script
├── predict.py              # Interactive prediction interface
├── confusion_matrix.png    # Model performance visualization
├── models/                 # Saved models directory
│   ├── text_classifier.pkl     # Trained classification model
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
└── README.md               # Project documentation
🔥 Quick Start Examples
Training the Model
bash
python train_model.py
What happens:

Downloads SMS Spam Collection dataset

Preprocesses and analyzes the data

Trains Naive Bayes and Logistic Regression models

Compares performance and selects the best model

Saves the trained model to models/ folder

Generates confusion matrix visualization

Making Predictions
bash
python predict.py
Then test with these examples:

text
"WINNER!! You've won $1000! Call now to claim." → SPAM (97.9% confidence)
"Hey, are we meeting for lunch tomorrow?" → HAM (98.5% confidence)
"Free iPhone! Text YES to claim" → SPAM (95.2% confidence)
"Can you pick up milk on your way home?" → HAM (99.1% confidence)
📈 Model Comparison
Model	Accuracy	Precision	Recall	F1-Score
Naive Bayes	97.76%	0.98	0.98	0.98
Logistic Regression	97.40%	0.97	0.97	0.97
🗃️ Dataset
SMS Spam Collection Dataset

Source: UCI Machine Learning Repository

Total samples: 5,572 messages

Ham messages: 4,825 (86.6%)

Spam messages: 747 (13.4%)

Format: TSV file with label and text columns

🔧 Technical Details
Preprocessing
Text lowercasing

Stop word removal

Special character cleaning

TF-IDF vectorization with 1,000 features

Algorithms
Naive Bayes: Multinomial variant optimized for text classification

Logistic Regression: L2 regularization with increased max iterations

Evaluation
80/20 train-test split

Stratified sampling to maintain class distribution

Comprehensive classification metrics

Visual confusion matrix

🎓 Learning Outcomes
This project demonstrates:

End-to-end machine learning pipeline development

Text preprocessing and feature engineering

Model selection and hyperparameter tuning

Performance evaluation and visualization

Model deployment and inference

🤝 Contributing
This project was developed as part of the Linkplus IT AI Internship Challenge. Feel free to fork and extend the functionality.

📄 License
This project is open source and available under the MIT License.

Developed by: Festim Gashi
Contact: festimi2005gashi@gmail.com
