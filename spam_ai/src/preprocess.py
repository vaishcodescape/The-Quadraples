"""
Message Preprocessing Module

Handles text normalization for spam detection:
- Lowercasing
- URL normalization
- Number normalization
- Noise removal
"""

import re
from typing import Optional


def preprocess_message(text: str, metadata: Optional[dict] = None) -> dict:
    """
    Preprocess a raw text message for spam detection.
    
    Args:
        text: Raw message text
        metadata: Optional metadata (sender, time, channel)
    
    Returns:
        Dictionary with processed text and extracted features
    """
    original_text = text
    
    # Store original features before normalization
    url_count = len(re.findall(r'https?://\S+|www\.\S+', text))
    has_urls = url_count > 0
    
    # Count digits before normalization
    digit_count = sum(c.isdigit() for c in text)
    
    # Calculate capital letter ratio
    alpha_chars = [c for c in text if c.isalpha()]
    capital_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if alpha_chars else 0
    
    # Lowercase
    processed = text.lower()
    
    # Normalize URLs
    processed = re.sub(r'https?://\S+|www\.\S+', '<URL>', processed)
    
    # Normalize phone numbers (various formats)
    processed = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '<PHONE>', processed)
    
    # Normalize money amounts
    processed = re.sub(r'[\$£€₹]\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd|inr|rupees?)', '<MONEY>', processed)
    
    # Normalize remaining numbers
    processed = re.sub(r'\b\d+\b', '<NUM>', processed)
    
    # Remove extra whitespace
    processed = re.sub(r'\s+', ' ', processed).strip()
    
    # Remove special characters but keep meaningful punctuation
    processed = re.sub(r'[^\w\s<>!?.,]', '', processed)
    
    return {
        'original_text': original_text,
        'processed_text': processed,
        'features': {
            'message_length': len(original_text),
            'digit_count': digit_count,
            'url_count': url_count,
            'has_urls': has_urls,
            'capital_ratio': round(capital_ratio, 4),
            'word_count': len(original_text.split())
        },
        'metadata': metadata or {}
    }


def batch_preprocess(messages: list[str]) -> list[dict]:
    """
    Preprocess multiple messages.
    
    Args:
        messages: List of raw message texts
    
    Returns:
        List of preprocessed message dictionaries
    """
    return [preprocess_message(msg) for msg in messages]


if __name__ == "__main__":
    # Test preprocessing
    test_messages = [
        "URGENT! You won $1,000,000! Click http://fakeprize.com NOW! Call +91-9876543210",
        "Hey, are we still meeting at 3pm tomorrow?",
        "Your OTP is 123456. Do not share with anyone. -Bank",
    ]
    
    for msg in test_messages:
        result = preprocess_message(msg)
        print(f"Original: {result['original_text']}")
        print(f"Processed: {result['processed_text']}")
        print(f"Features: {result['features']}")
        print("-" * 50)
