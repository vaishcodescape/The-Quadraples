"""
Spam Detection Training Module

Trains a TF-IDF + Logistic Regression classifier for binary spam detection.
Optimized for CPU-only execution.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_message


# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')


def load_or_create_dataset() -> pd.DataFrame:
    """
    Load SMS Spam Collection dataset or create sample data.
    
    The real dataset should be placed at data/spam.csv with columns:
    - label: 'spam' or 'ham'
    - text: message content
    """
    csv_path = os.path.join(DATA_DIR, 'spam.csv')
    
    if os.path.exists(csv_path):
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                # Handle common column names from UCI dataset
                if 'v1' in df.columns and 'v2' in df.columns:
                    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
                    df = df[['label', 'text']]
                print(f"Loaded {len(df)} messages from {csv_path}")
                return df
            except Exception as e:
                continue
        print(f"Warning: Could not read {csv_path}. Using sample data.")
    
    # Create comprehensive sample dataset if no file exists
    print("Creating sample training dataset...")
    sample_data = {
        'label': [],
        'text': []
    }
    
    # Spam messages (varied examples)
    spam_messages = [
        "URGENT! You have won $1,000,000! Click http://fakeprize.com to claim NOW!",
        "Congratulations! You've been selected for a free iPhone. Visit bit.ly/freeiphone",
        "Your bank account has been compromised. Verify now: http://fake-bank.com",
        "WINNER! You've won a $500 gift card. Claim here: http://giftcard-scam.com",
        "Hot singles in your area! Click now: http://dating-scam.com",
        "Make $5000 daily from home! No experience needed. Reply YES",
        "Your OTP is 123456. Share this code to verify your account urgently",
        "Dear customer, your SBI account will be blocked. Update KYC: sbi-update.fake.com",
        "You've been pre-approved for a loan of $50,000! Act now before offer expires",
        "ALERT: Suspicious activity on your card. Call +91-9999999999 immediately",
        "Free entry to win a Mercedes! Just send your details to claim",
        "Your Amazon order #12345 is stuck. Verify payment: amazon-verify.fake.com",
        "Congratulations! You've won Rs.50,00,000 in KBC lottery. Contact now!",
        "URGENT: Your WhatsApp will expire. Click to renew: whatsapp-renew.com",
        "You've received $1000 in Bitcoin! Claim now at crypto-prize.com",
        "Job offer: Earn $3000/week working from home. No interview needed!",
        "Your Netflix subscription failed. Update payment: netflix-billing.fake.com",
        "WINNER ALERT! You've won a free vacation to Maldives. Claim now!",
        "Exclusive investment opportunity! Double your money in 30 days guaranteed",
        "Your insurance policy is expiring! Renew now to avoid penalties",
        "FREE: Get 100GB data instantly! Just click and verify your number",
        "Dear valued customer, verify your HDFC account: hdfc-verify.fake.com",
        "Last chance! 90% off on all products. Limited time only!",
        "You've been charged $499. If not you, call immediately: 1-800-FAKE",
        "CONGRATS! Your number selected for Rs.25 lakhs prize. Send Aadhar to claim",
        "Warning: Your computer is infected! Call Microsoft support: 1-800-SCAM",
        "Exclusive offer: iPhone 15 at 99% discount! Only 10 left!",
        "Your payment of $299 is pending. Complete now or face legal action",
        "Hi, I'm a Nigerian prince. I need your help to transfer $10 million",
        "URGENT: IRS notice - Pay tax dues immediately or face arrest",
    ]
    
    # Ham messages (normal legitimate messages)
    ham_messages = [
        "Hey, are we still meeting for coffee tomorrow at 3pm?",
        "Can you pick up some groceries on your way home?",
        "Happy birthday! Hope you have a great day!",
        "The meeting has been rescheduled to 4pm today",
        "Thanks for dinner last night. It was delicious!",
        "Don't forget to submit the report by Friday",
        "I'll be there in 10 minutes",
        "Can you send me the presentation slides?",
        "Movie tonight? Let me know if you're interested",
        "Your package has been delivered to your doorstep",
        "The weather looks great for the weekend trip",
        "Please review the attached document when you get a chance",
        "Running late, traffic is terrible today",
        "Loved the photos from your vacation!",
        "Let's catch up over lunch this week",
        "The project deadline has been extended by a week",
        "Can you help me move this Saturday?",
        "Your order has shipped and will arrive by Monday",
        "Remember to call mom on her birthday tomorrow",
        "Great job on the presentation today!",
        "I'll send you the address for the party",
        "What time does the movie start?",
        "Thanks for your help with the project",
        "See you at the gym tomorrow morning",
        "The kids are doing well at school",
        "Can you recommend a good restaurant nearby?",
        "Happy anniversary! Wishing you many more years together",
        "I've attached the invoice for your records",
        "Let me know when you're free to discuss the proposal",
        "The doctor's appointment is confirmed for Tuesday",
        "How was your weekend? Did you enjoy the concert?",
        "Please bring your ID to the meeting tomorrow",
        "The internet is down, I'm using mobile data",
        "I've updated the shared document with the changes",
        "Can you water my plants while I'm on vacation?",
        "Your verification code is 1234. Do not share it.",
        "Your package from Amazon has been delivered.",
        "Your lightning AI veification code is: 996214",
        "Your OTP for transaction is 4521. Valid for 10 mins.",
        "Use 8829 as your login OTP. Don't share with anyone.",
        "Your cab will arrive in 5 mins. OTP is 1234.",
        "Payment of Rs. 500 received from Rahul.",
        "Your appointment is confirmed for 4pm today.",
        "A/c *1464 Debited for Rs:1500 on 05-02-2026. Avl Bal Rs:210.",
        "Rs. 2000 credited to A/c XX8923 via UPI.",
        "Bill payment for Electricity of Rs 1400 is successful.",
        "Transaction ID 892374823 for Rs 550 is successful.",
        "Your request for cheque book has been received. Ref: SR12389",
        "Dear Customer, your mobile bill of Rs 499 is generated.",
        "Refund of Rs 200 initiated for order #12345.",
    ]
    
    for msg in spam_messages:
        sample_data['label'].append('spam')
        sample_data['text'].append(msg)
    
    for msg in ham_messages:
        sample_data['label'].append('ham')
        sample_data['text'].append(msg)
    
    df = pd.DataFrame(sample_data)
    
    # Save the sample data
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Sample dataset saved to {csv_path}")
    
    return df


def train_spam_detector():
    """Train the spam detection model."""
    print("=" * 60)
    print("SPAM DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_or_create_dataset()
    
    # Preprocess messages
    print("\nPreprocessing messages...")
    processed_texts = []
    for text in df['text']:
        result = preprocess_message(text)
        processed_texts.append(result['processed_text'])
    
    df['processed_text'] = processed_texts
    
    # Convert labels to binary
    df['spam'] = (df['label'] == 'spam').astype(int)
    
    # Split data
    X = df['processed_text']
    y = df['spam']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Spam ratio (train): {y_train.mean():.2%}")
    
    # Create TF-IDF vectorizer
    print("\nTraining TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression classifier...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=1.0
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    y_pred = model.predict(X_test_tfidf)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation
    print("\nCross-validation scores:")
    cv_scores = cross_val_score(model, vectorizer.transform(X), y, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save models
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    vectorizer_path = os.path.join(MODEL_DIR, 'spam_vectorizer.joblib')
    model_path = os.path.join(MODEL_DIR, 'spam_detector.joblib')
    
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)
    
    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"Model saved to: {model_path}")
    
    print("\n✓ Training complete!")
    
    return vectorizer, model


def predict_spam(text: str, vectorizer=None, model=None) -> dict:
    """
    Predict whether a message is spam.
    
    Args:
        text: Message text
        vectorizer: TF-IDF vectorizer (loads from disk if None)
        model: Trained model (loads from disk if None)
    
    Returns:
        Dictionary with prediction results
    """
    if vectorizer is None or model is None:
        vectorizer_path = os.path.join(MODEL_DIR, 'spam_vectorizer.joblib')
        model_path = os.path.join(MODEL_DIR, 'spam_detector.joblib')
        
        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Models not found. Run training first.")
        
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
    
    # Preprocess
    processed = preprocess_message(text)
    
    # Vectorize
    X = vectorizer.transform([processed['processed_text']])
    
    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    return {
        'is_spam': bool(prediction),
        'confidence': float(probabilities[1] if prediction else probabilities[0]),
        'spam_probability': float(probabilities[1]),
        'ham_probability': float(probabilities[0]),
    }


if __name__ == "__main__":
    train_spam_detector()
    
    # Test predictions
    print("\n" + "=" * 60)
    print("TEST PREDICTIONS")
    print("=" * 60)
    
    test_messages = [
        "URGENT! You won $1,000,000! Click now!",
        "Hey, are we meeting tomorrow?",
        "Your bank account is suspended. Verify now: fake-bank.com",
        "Happy birthday! Hope you have a wonderful day!",
    ]
    
    for msg in test_messages:
        result = predict_spam(msg)
        print(f"\nMessage: {msg}")
        print(f"  Spam: {result['is_spam']} (confidence: {result['confidence']:.2%})")
