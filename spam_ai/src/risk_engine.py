"""
Risk Scoring Engine

Computes comprehensive risk scores (0-100) based on:
- Spam detection confidence
- Scam type severity
- Urgency indicators
- Persona vulnerability
- Threat intelligence factors
"""

from typing import Optional


# Scam type severity weights (higher = more dangerous)
SCAM_SEVERITY = {
    'bank_phishing': 0.95,
    'otp_scam': 0.90,
    'crypto_scam': 0.85,
    'loan_scam': 0.80,
    'lottery_scam': 0.75,
    'job_fraud': 0.70,
    'other': 0.60
}


# Risk level thresholds
RISK_LEVELS = {
    (0, 25): 'low',
    (25, 50): 'medium',
    (50, 75): 'high',
    (75, 100): 'critical'
}


# Recommended actions based on risk
RECOMMENDATIONS = {
    'low': {
        'action': 'monitor',
        'description': 'Monitor but no immediate action needed',
        'details': [
            'Keep the message for reference',
            'Be cautious with similar future messages',
            'No urgent action required'
        ]
    },
    'medium': {
        'action': 'caution',
        'description': 'Exercise caution and verify',
        'details': [
            'Do not click any links',
            'Verify sender through official channels',
            'Do not share personal information',
            'Consider reporting to spam filters'
        ]
    },
    'high': {
        'action': 'block_and_report',
        'description': 'Block sender and report as spam',
        'details': [
            'Immediately block the sender',
            'Report to spam/scam reporting services',
            'Do not respond under any circumstances',
            'Alert family members about similar scams',
            'Check your accounts for unauthorized activity'
        ]
    },
    'critical': {
        'action': 'block_report_alert',
        'description': 'Critical threat - take immediate action',
        'details': [
            'Block sender immediately',
            'Report to cybercrime authorities',
            'Check all financial accounts immediately',
            'Change passwords if any were shared',
            'Enable 2FA on all accounts',
            'Consider freezing credit if personal data was shared',
            'Alert family and friends about this scam'
        ]
    }
}


def get_risk_level(score: float) -> str:
    """Convert numeric score to risk level string."""
    for (low, high), level in RISK_LEVELS.items():
        if low <= score < high:
            return level
    return 'critical' if score >= 100 else 'low'


def calculate_risk_score(
    spam_confidence: float,
    scam_type: Optional[str] = None,
    urgency_score: float = 0,
    threat_indicator_count: int = 0,
    persona_vulnerability: float = 0.5,
    has_urls: bool = False,
    has_phone_numbers: bool = False,
    has_credential_request: bool = False
) -> dict:
    """
    Calculate comprehensive risk score.
    
    Args:
        spam_confidence: Confidence from spam detection (0-1)
        scam_type: Detected scam type
        urgency_score: Urgency indicator score (0-1)
        threat_indicator_count: Number of threat indicators found
        persona_vulnerability: Victim vulnerability score (0-1)
        has_urls: Whether message contains URLs
        has_phone_numbers: Whether message contains phone numbers
        has_credential_request: Whether message requests credentials
    
    Returns:
        Dictionary with risk score and recommendations
    """
    # Base score from spam confidence (0-40 points)
    base_score = spam_confidence * 40
    
    # Scam type severity bonus (0-25 points)
    scam_severity = SCAM_SEVERITY.get(scam_type, 0.5) if scam_type else 0
    scam_score = scam_severity * 25
    
    # Urgency indicators (0-10 points)
    urgency_score_points = min(urgency_score, 1) * 10
    
    # Threat indicators (0-15 points)
    threat_score = min(threat_indicator_count / 5, 1) * 15
    
    # Contact method risk (0-10 points)
    contact_risk = 0
    if has_urls:
        contact_risk += 5
    if has_phone_numbers:
        contact_risk += 3
    if has_credential_request:
        contact_risk += 7
    contact_risk = min(contact_risk, 10)
    
    # Calculate raw score
    raw_score = base_score + scam_score + urgency_score_points + threat_score + contact_risk
    
    # Apply persona vulnerability modifier
    # Higher vulnerability increases score by up to 20%
    vulnerability_modifier = 1 + (persona_vulnerability - 0.5) * 0.4
    final_score = raw_score * vulnerability_modifier
    
    # Clamp to 0-100
    final_score = max(0, min(100, final_score))
    
    # Get risk level
    risk_level = get_risk_level(final_score)
    
    # Get recommendations
    recommendation = RECOMMENDATIONS.get(risk_level, RECOMMENDATIONS['medium'])
    
    # Build detailed breakdown
    score_breakdown = {
        'base_spam_score': round(base_score, 2),
        'scam_severity_score': round(scam_score, 2),
        'urgency_score': round(urgency_score_points, 2),
        'threat_indicator_score': round(threat_score, 2),
        'contact_risk_score': round(contact_risk, 2),
        'raw_total': round(raw_score, 2),
        'vulnerability_modifier': round(vulnerability_modifier, 2),
        'final_score': round(final_score, 2)
    }
    
    return {
        'risk_score': round(final_score),
        'risk_level': risk_level,
        'score_breakdown': score_breakdown,
        'recommended_action': recommendation['action'],
        'action_description': recommendation['description'],
        'action_details': recommendation['details'],
        'factors': {
            'spam_detected': spam_confidence > 0.5,
            'spam_confidence': round(spam_confidence, 4),
            'scam_type': scam_type,
            'scam_severity': round(scam_severity, 2) if scam_type else None,
            'has_urls': has_urls,
            'has_phone_numbers': has_phone_numbers,
            'has_credential_request': has_credential_request,
            'threat_indicator_count': threat_indicator_count,
            'persona_vulnerability': round(persona_vulnerability, 2)
        }
    }


def assess_message_risk(
    analysis_result: dict,
    persona: str = 'working_professional'
) -> dict:
    """
    Assess risk from a complete message analysis result.
    
    Args:
        analysis_result: Result from full message analysis
        persona: Assumed victim persona
    
    Returns:
        Complete risk assessment
    """
    # Default persona vulnerabilities
    PERSONA_VULNERABILITY = {
        'elderly': 0.85,
        'student': 0.45,
        'working_professional': 0.35,
        'small_business_owner': 0.55
    }
    
    vulnerability = PERSONA_VULNERABILITY.get(persona, 0.5)
    
    # Extract relevant fields from analysis
    spam_confidence = analysis_result.get('spam_confidence', 0)
    scam_type = analysis_result.get('scam_type')
    
    # Get feature data
    features = analysis_result.get('features', {})
    urgency = features.get('urgency_normalized', 0)
    
    # Get threat intelligence
    intel = analysis_result.get('threat_intelligence', {})
    threat_count = intel.get('threat_indicator_count', 0)
    indicators = intel.get('threat_indicators', {})
    
    return calculate_risk_score(
        spam_confidence=spam_confidence,
        scam_type=scam_type,
        urgency_score=urgency,
        threat_indicator_count=threat_count,
        persona_vulnerability=vulnerability,
        has_urls=indicators.get('has_urls', False),
        has_phone_numbers=indicators.get('has_phones', False),
        has_credential_request=indicators.get('has_credential_request', False)
    )


if __name__ == "__main__":
    print("=" * 60)
    print("RISK SCORING ENGINE TEST")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'High Risk - Bank Phishing',
            'spam_confidence': 0.95,
            'scam_type': 'bank_phishing',
            'urgency_score': 0.8,
            'threat_indicator_count': 4,
            'persona_vulnerability': 0.85,
            'has_urls': True,
            'has_credential_request': True
        },
        {
            'name': 'Medium Risk - Job Scam',
            'spam_confidence': 0.75,
            'scam_type': 'job_fraud',
            'urgency_score': 0.5,
            'threat_indicator_count': 2,
            'persona_vulnerability': 0.45,
            'has_urls': True,
            'has_credential_request': False
        },
        {
            'name': 'Low Risk - Borderline Ham',
            'spam_confidence': 0.3,
            'scam_type': None,
            'urgency_score': 0.1,
            'threat_indicator_count': 0,
            'persona_vulnerability': 0.35,
            'has_urls': False,
            'has_credential_request': False
        }
    ]
    
    for scenario in scenarios:
        name = scenario.pop('name')
        result = calculate_risk_score(**scenario)
        
        print(f"\n{'-'*60}")
        print(f"Scenario: {name}")
        print(f"{'-'*60}")
        print(f"Risk Score: {result['risk_score']}/100")
        print(f"Risk Level: {result['risk_level'].upper()}")
        print(f"Action: {result['recommended_action']}")
        print(f"\nScore Breakdown:")
        for key, value in result['score_breakdown'].items():
            print(f"  - {key}: {value}")
        print(f"\nRecommendations:")
        for detail in result['action_details'][:3]:
            print(f"  • {detail}")
