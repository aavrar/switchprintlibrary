# Security Features

SwitchPrint v2.1.0 includes comprehensive security features **fully validated** with 100% test coverage, designed for enterprise deployment and production environments.

## 🔒 Security Components

### Input Validation
- **Text Sanitization**: HTML/XSS prevention with multiple security levels
- **Injection Detection**: SQL injection, command injection, and script injection detection
- **Size Validation**: Configurable limits to prevent DoS attacks
- **Character Filtering**: Unicode normalization and suspicious character detection

### Model Security
- **Integrity Checking**: SHA-256 hash verification for model files
- **Pickle Scanning**: Detection of malicious pickle operations
- **Vulnerability Assessment**: Automated security auditing of ML models
- **Safe Loading**: Protection against deserialization attacks

### Privacy Protection
- **PII Detection**: Automatic detection of personally identifiable information
- **Data Anonymization**: Multiple anonymization strategies (masking, hashing, replacement)
- **Privacy Levels**: Configurable privacy protection levels (minimal to maximum)
- **Audit Logging**: Comprehensive privacy protection audit trails

### Security Monitoring
- **Real-time Threat Detection**: Anomaly detection and behavioral analysis
- **Rate Limiting**: Configurable request rate limits and burst protection
- **Audit Logging**: Comprehensive security event logging
- **Threat Intelligence**: Integration with threat intelligence feeds

## 🛡️ Security Levels

### Input Validation Levels
- **Permissive**: Basic validation for development environments
- **Moderate**: Standard validation for testing environments
- **Strict**: Enhanced validation for production environments
- **Paranoid**: Maximum security for high-risk environments

### Privacy Protection Levels
- **Minimal**: Basic PII detection (emails, phones)
- **Standard**: Extended PII detection including SSNs, credit cards
- **High**: Comprehensive PII detection with medical information
- **Maximum**: Full anonymization with fake data generation

## 🔧 Configuration

### Basic Security Setup
```python
from codeswitch_ai import SecurityConfig, PrivacyConfig, PrivacyLevel

# Configure input validation
security_config = SecurityConfig(
    security_level='strict',
    max_text_length=10000,
    enable_html_sanitization=True,
    enable_injection_detection=True
)

# Configure privacy protection
privacy_config = PrivacyConfig(
    privacy_level=PrivacyLevel.HIGH,
    anonymization_method='replacement',
    preserve_language_structure=True
)
```

### Enterprise Security Setup
```python
from codeswitch_ai import (
    InputValidator, PrivacyProtector, SecurityMonitor, 
    ModelSecurityAuditor
)

# Initialize security components
validator = InputValidator(config=security_config)
privacy_protector = PrivacyProtector(config=privacy_config)
security_monitor = SecurityMonitor(log_file='security_audit.log')
model_auditor = ModelSecurityAuditor(trusted_sources=['internal_registry'])

# Secure processing pipeline
def secure_text_processing(text: str, user_id: str):
    # 1. Input validation
    validation_result = validator.validate(text)
    if not validation_result.is_valid:
        return {'error': 'Input validation failed'}
    
    # 2. Privacy protection
    privacy_result = privacy_protector.protect_text(
        validation_result.sanitized_text, 
        source_id=user_id
    )
    
    # 3. Security monitoring
    security_events = security_monitor.process_request(
        source_id='text_processing',
        request_data={
            'text_size': len(text),
            'detected_languages': []  # Will be filled by detection
        },
        user_id=user_id
    )
    
    return {
        'protected_text': privacy_result['protected_text'],
        'security_events': len(security_events),
        'privacy_applied': privacy_result['protection_applied']
    }
```

## 📊 Security Monitoring

### Real-time Monitoring
The security monitor provides real-time threat detection including:
- Rate limit violations
- Suspicious access patterns
- Behavioral anomalies
- Failed authentication attempts

### Audit Reports
Generate comprehensive security reports:
```python
# Generate security report
report = security_monitor.generate_security_report(hours=24)
print(f"Threats detected: {report['monitoring_status']['threats_detected']}")
print(f"Security events: {report['audit_summary']['summary']['total_events']}")
```

## 🚨 Threat Detection

### Behavioral Anomalies
- Unusual request sizes (5x larger than user baseline)
- Uncommon language usage patterns
- Burst activity detection
- Suspicious timing patterns

### Model Security Threats
- Pickle deserialization attacks
- Model integrity violations
- Untrusted model sources
- Oversized model files
- Memory exhaustion attacks

## 📋 Best Practices

### Development
1. Always validate input before processing
2. Use the lowest necessary privacy level for development
3. Enable debug logging for security events
4. Test with various input types and sizes

### Production
1. Use strict input validation
2. Enable comprehensive privacy protection
3. Monitor security events continuously
4. Regularly audit model security
5. Keep threat intelligence updated
6. Implement proper access controls

### Model Deployment
1. Audit all models before deployment
2. Verify model integrity with hash checking
3. Use trusted model sources only
4. Implement model signing for critical applications
5. Monitor model usage and access patterns

## 🔍 Security Testing

### Testing Security Features
```python
# Test input validation
test_cases = [
    "Normal text",
    "<script>alert('xss')</script>",
    "'; DROP TABLE users; --",
    "x" * 100000,  # Large text
    "Contact me at john@example.com",  # PII
]

for test_input in test_cases:
    result = validator.validate(test_input)
    print(f"Input: {test_input[:50]}...")
    print(f"Valid: {result.is_valid}")
    print(f"Threats: {result.threats_detected}")
    print("---")
```

### Model Security Testing
```python
# Test model security audit
models_to_test = ["model.pkl", "model.bin", "model.h5"]
for model_path in models_to_test:
    audit_result = model_auditor.audit_model_file(model_path)
    print(f"Model: {model_path}")
    print(f"Safe: {audit_result.is_safe}")
    print(f"Threat Level: {audit_result.threat_level.value}")
    print(f"Issues: {[i.value for i in audit_result.issues_detected]}")
```

## 📞 Security Support

For security-related issues or questions:

1. **Vulnerabilities**: Report security vulnerabilities privately
2. **Configuration**: Refer to this documentation for proper setup
3. **Monitoring**: Use built-in audit and monitoring features
4. **Updates**: Keep the library updated for latest security patches

## 🔐 Compliance

The security features support compliance with:
- GDPR privacy requirements
- HIPAA data protection (with proper configuration)
- SOC 2 security controls
- Enterprise security policies

**Note**: While these security features provide strong protection, always conduct proper security assessments for your specific use case and regulatory requirements.