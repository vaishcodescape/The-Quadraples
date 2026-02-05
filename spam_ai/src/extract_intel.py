"""
Threat Intelligence Extraction Module

Extracts security-relevant information from messages:
- URLs (regex)
- Phone numbers (regex)
- Money amounts (regex)
- Organization names (spaCy NER)
- Action phrases (rule-based)
"""

import re
from typing import Optional

# Try to import spacy, handle gracefully if not installed
try:
    import spacy
    NLP = None  # Lazy load
except ImportError:
    spacy = None
    NLP = None


def get_nlp():
    """Lazy load spaCy model to save memory."""
    global NLP
    if NLP is None and spacy is not None:
        try:
            NLP = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")
            NLP = False
    return NLP if NLP else None


# Regex patterns for extraction
URL_PATTERN = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')
# Indian mobile number focused pattern to avoid false positives like "5000-10000"
# Matches: +91 9876543210, +919876543210, 91-9876543210, 9876543210
PHONE_PATTERN = re.compile(
    r'(?:'
    r'\+?91[\s.-]?[6-9]\d{9}'  # Indian with country code
    r'|'
    r'(?<![0-9\-])[6-9]\d{9}(?![0-9\-])'  # Standalone 10-digit Indian mobile (not part of range)
    r')'
)
MONEY_PATTERN = re.compile(r'[\$£€₹]\s*[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd|inr|rupees?|pounds?|euros?)', re.IGNORECASE)
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# Action phrases indicating scam behavior
ACTION_PHRASES = {
    'credential_request': [
        'enter your password', 'provide your otp', 'share your pin',
        'send your cvv', 'confirm your card', 'verify your account',
        'update your details', 'enter your bank', 'provide account number'
    ],
    'urgency': [
        'act now', 'immediately', 'urgent', 'expires today',
        'last chance', 'final warning', 'account suspended',
        'verify within', 'confirm within', 'respond immediately'
    ],
    'reward_claim': [
        'claim your prize', 'collect your winnings', 'you have won',
        'congratulations', 'selected as winner', 'lucky winner',
        'claim reward', 'redeem now', 'bonus waiting'
    ],
    'threat': [
        'account will be blocked', 'legal action', 'police complaint',
        'account suspended', 'unauthorized access', 'security breach',
        'immediate action required', 'failure to respond'
    ]
}


def extract_urls(text: str) -> list[str]:
    """Extract all URLs from text."""
    urls = URL_PATTERN.findall(text)
    # Clean up URLs
    cleaned = []
    for url in urls:
        # Remove trailing punctuation
        url = url.rstrip('.,;:!?)')
        if url:
            cleaned.append(url)
    return list(set(cleaned))


def extract_phone_numbers(text: str) -> list[str]:
    """Extract phone numbers from text."""
    phones = PHONE_PATTERN.findall(text)
    # Filter out short matches (likely not phone numbers)
    return list(set(p for p in phones if len(re.sub(r'\D', '', p)) >= 7))


def extract_money_amounts(text: str) -> list[str]:
    """Extract money amounts from text."""
    amounts = MONEY_PATTERN.findall(text)
    return list(set(amounts))


def extract_emails(text: str) -> list[str]:
    """Extract email addresses from text."""
    emails = EMAIL_PATTERN.findall(text)
    return list(set(emails))


def extract_organizations(text: str) -> list[str]:
    """Extract organization names using spaCy NER."""
    nlp = get_nlp()
    if nlp is None:
        return []
    
    doc = nlp(text)
    orgs = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            orgs.append(ent.text)
    return list(set(orgs))


def extract_action_phrases(text: str) -> dict:
    """Extract action phrases by category using rule-based matching."""
    text_lower = text.lower()
    found_phrases = {}
    
    for category, phrases in ACTION_PHRASES.items():
        matches = [phrase for phrase in phrases if phrase in text_lower]
        if matches:
            found_phrases[category] = matches
    
    return found_phrases


def extract_threat_intelligence(text: str) -> dict:
    """
    Extract all threat intelligence from a message.
    
    Args:
        text: Raw message text
    
    Returns:
        Dictionary containing all extracted threat indicators
    """
    urls = extract_urls(text)
    phones = extract_phone_numbers(text)
    money = extract_money_amounts(text)
    emails = extract_emails(text)
    orgs = extract_organizations(text)
    action_phrases = extract_action_phrases(text)
    
    # Calculate threat indicators
    has_suspicious_url = any(
        'bit.ly' in url.lower() or 
        'tinyurl' in url.lower() or
        len(url) < 20 or
        not any(trusted in url.lower() for trusted in ['google', 'microsoft', 'apple', 'amazon', 'facebook'])
        for url in urls
    ) if urls else False
    
    threat_indicators = {
        'has_urls': len(urls) > 0,
        'has_phones': len(phones) > 0,
        'has_money_mentions': len(money) > 0,
        'has_credential_request': 'credential_request' in action_phrases,
        'has_urgency': 'urgency' in action_phrases,
        'has_reward_claim': 'reward_claim' in action_phrases,
        'has_threat': 'threat' in action_phrases,
        'has_suspicious_url': has_suspicious_url,
    }
    
    # Count total threat indicators
    threat_count = sum(1 for v in threat_indicators.values() if v)
    
    return {
        'urls': urls,
        'phones': phones,
        'money_amounts': money,
        'emails': emails,
        'organizations': orgs,
        'action_phrases': action_phrases,
        'threat_indicators': threat_indicators,
        'threat_indicator_count': threat_count,
    }


if __name__ == "__main__":
    # Test threat intelligence extraction
    test_messages = [
        "URGENT! Your SBI account will be blocked. Click http://sbi-verify.fake.com to verify. Call +91-9876543210 for help. You won $50,000!",
        "Hey, are we still meeting at 3pm tomorrow?",
        "Dear customer, your OTP is 123456. Never share this with anyone. -HDFC Bank",
        "Congratulations! You've been selected for a $10,000 prize. Claim at http://bit.ly/prize123. Provide your bank details to receive payment.",
    ]
    
    for msg in test_messages:
        print(f"Message: {msg[:60]}...")
        intel = extract_threat_intelligence(msg)
        print(f"  URLs: {intel['urls']}")
        print(f"  Phones: {intel['phones']}")
        print(f"  Money: {intel['money_amounts']}")
        print(f"  Orgs: {intel['organizations']}")
        print(f"  Actions: {intel['action_phrases']}")
        print(f"  Threat Count: {intel['threat_indicator_count']}")
        print("-" * 60)
