"""
Feature Engineering Module

Extracts metadata features to improve spam detection accuracy:
- Message length and word count
- Digit and URL counts
- Urgency keyword scoring
- Capital letter ratio
- Repetition patterns
"""

import re
from typing import Optional


# Urgency keywords commonly found in spam/scam messages
URGENCY_KEYWORDS = [
    'urgent', 'immediately', 'now', 'hurry', 'quick', 'fast',
    'limited', 'expire', 'deadline', 'asap', 'act now', 'don\'t wait',
    'last chance', 'final notice', 'warning', 'alert', 'important',
    'attention', 'action required', 'verify now', 'confirm now',
    'click here', 'call now', 'respond now', 'today only'
]

# Action phrases that indicate scam behavior
ACTION_PHRASES = [
    'click here', 'click the link', 'click below', 'tap here',
    'call this number', 'call now', 'call immediately',
    'send money', 'transfer funds', 'pay now', 'payment required',
    'verify your account', 'confirm your identity', 'update your information',
    'enter your', 'provide your', 'share your', 'send your',
    'otp', 'pin', 'password', 'cvv', 'card number',
    'won', 'winner', 'congratulations', 'selected', 'chosen',
    'claim your', 'collect your', 'redeem your'
]


def extract_features(text: str, preprocessed: Optional[dict] = None) -> dict:
    """
    Extract all relevant features from a message.
    
    Args:
        text: Original message text
        preprocessed: Optional preprocessed data (to avoid recomputation)
    
    Returns:
        Dictionary of extracted features
    """
    text_lower = text.lower()
    
    # Basic text statistics
    message_length = len(text)
    word_count = len(text.split())
    digit_count = sum(c.isdigit() for c in text)
    
    # URL detection
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    url_count = len(urls)
    has_urls = url_count > 0
    
    # Capital letter analysis
    alpha_chars = [c for c in text if c.isalpha()]
    capital_count = sum(1 for c in alpha_chars if c.isupper())
    capital_ratio = capital_count / len(alpha_chars) if alpha_chars else 0
    
    # Urgency scoring
    urgency_score = sum(1 for keyword in URGENCY_KEYWORDS if keyword in text_lower)
    urgency_normalized = min(urgency_score / 5, 1.0)  # Normalize to 0-1
    
    # Action phrase detection
    action_phrases_found = [phrase for phrase in ACTION_PHRASES if phrase in text_lower]
    action_score = len(action_phrases_found)
    
    # Exclamation and question marks (often excessive in spam)
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Repetition analysis (repeated characters like "URGENTTT" or "freeeee")
    repetition_pattern = re.findall(r'(.)\1{2,}', text_lower)
    repetition_score = len(repetition_pattern)
    
    # Special character ratio
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_char_ratio = special_chars / len(text) if text else 0
    
    # Sentence structure
    sentence_count = len(re.split(r'[.!?]+', text))
    avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count else 0
    
    # Phone number detection
    phone_patterns = re.findall(r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', text)
    phone_count = len(phone_patterns)
    
    # Money amount detection
    money_patterns = re.findall(r'[\$£€₹]\s*\d+(?:,\d{3})*(?:\.\d{2})?', text)
    money_count = len(money_patterns)
    
    return {
        # Basic statistics
        'message_length': message_length,
        'word_count': word_count,
        'digit_count': digit_count,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 2),
        
        # URL features
        'url_count': url_count,
        'has_urls': int(has_urls),
        
        # Character analysis
        'capital_ratio': round(capital_ratio, 4),
        'special_char_ratio': round(special_char_ratio, 4),
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        
        # Spam indicators
        'urgency_score': urgency_score,
        'urgency_normalized': round(urgency_normalized, 4),
        'action_score': action_score,
        'action_phrases': action_phrases_found,
        'repetition_score': repetition_score,
        
        # Contact info
        'phone_count': phone_count,
        'money_count': money_count,
    }


def get_feature_vector(features: dict) -> list:
    """
    Convert feature dictionary to a numeric vector for ML models.
    
    Args:
        features: Dictionary of extracted features
    
    Returns:
        List of numeric feature values
    """
    return [
        features['message_length'],
        features['word_count'],
        features['digit_count'],
        features['url_count'],
        features['has_urls'],
        features['capital_ratio'],
        features['special_char_ratio'],
        features['exclamation_count'],
        features['urgency_normalized'],
        features['action_score'],
        features['repetition_score'],
        features['phone_count'],
        features['money_count'],
    ]


FEATURE_NAMES = [
    'message_length', 'word_count', 'digit_count', 'url_count', 'has_urls',
    'capital_ratio', 'special_char_ratio', 'exclamation_count',
    'urgency_normalized', 'action_score', 'repetition_score',
    'phone_count', 'money_count'
]


if __name__ == "__main__":
    # Test feature extraction
    test_messages = [
        "URGENT!!! You won $1,000,000! Click http://fakeprize.com NOW! Call +91-9876543210",
        "Hey, are we still meeting at 3pm tomorrow?",
        "Your OTP is 123456. Do not share with anyone. Verify your account NOW!",
    ]
    
    for msg in test_messages:
        features = extract_features(msg)
        print(f"Message: {msg[:50]}...")
        print(f"Urgency Score: {features['urgency_score']}")
        print(f"Action Score: {features['action_score']}")
        print(f"Capital Ratio: {features['capital_ratio']}")
        print(f"Feature Vector: {get_feature_vector(features)}")
        print("-" * 50)
