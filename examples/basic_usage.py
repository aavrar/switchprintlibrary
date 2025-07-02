#!/usr/bin/env python3
"""
Basic usage examples for the SwitchPrint Library v2.1.1
Demonstrates core detection capabilities with production-ready API
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codeswitch_ai import EnsembleDetector, FastTextDetector, TransformerDetector
from codeswitch_ai import PrivacyProtector, SecurityMonitor, InputValidator
from codeswitch_ai import ThresholdConfig, DetectionMode, ThresholdProfile


def basic_detection_example():
    """Basic language detection with ensemble detector."""
    print("🔍 Basic Language Detection")
    print("-" * 40)
    
    # Initialize ensemble detector with FastText + Transformer
    detector = EnsembleDetector(
        use_fasttext=True,
        use_transformer=False,  # Disable transformer for speed
        ensemble_strategy="weighted_average"
    )
    
    # Simple detection
    result = detector.detect_language("Hello, ¿cómo estás? I'm doing bien.")
    
    print(f"Text: 'Hello, ¿cómo estás? I'm doing bien.'")
    print(f"Detected languages: {result.detected_languages}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Method: {result.method}")
    print(f"Switch points: {getattr(result, 'switch_points', 'N/A')}")


def user_guided_detection():
    """Fast detection with FastText detector."""
    print("\n⚡ High-Speed Detection (FastText)")
    print("-" * 40)
    
    # Use FastText for maximum speed
    detector = FastTextDetector()
    
    # Fast detection
    result = detector.detect_language(
        "Main ghar ja raha hoon but I will come back"
    )
    
    print(f"Text: 'Main ghar ja raha hoon but I will come back'")
    print(f"Detected languages: {result.detected_languages}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Method: {result.method}")
    print(f"Probabilities: {getattr(result, 'probabilities', 'N/A')}")


def threshold_mode_analysis():
    """Demonstrate different threshold modes."""
    print("\n📊 Threshold Mode Analysis")
    print("-" * 40)
    
    # Test different threshold modes
    modes = [
        (DetectionMode.HIGH_PRECISION, "🎯 High Precision"),
        (DetectionMode.BALANCED, "⚖️ Balanced"),
        (DetectionMode.HIGH_RECALL, "🔍 High Recall")
    ]
    
    text = "Je suis tired today. Need some café."
    print(f"Text: '{text}'")
    print("\nMode comparison:")
    
    for mode, description in modes:
        config = ThresholdConfig(mode=mode)
        detector = EnsembleDetector(
            use_fasttext=True,
            use_transformer=False,
            threshold_config=config
        )
        
        result = detector.detect_language(text)
        print(f"  {description:15} | {', '.join(result.detected_languages):15} | {result.confidence:.1%}")


def multilingual_detection():
    """Demonstrate multilingual text detection."""
    print("\n🌍 Multilingual Text Detection")
    print("-" * 40)
    
    detector = EnsembleDetector(
        use_fasttext=True,
        use_transformer=False
    )
    
    test_cases = [
        "你好 hello world",
        "こんにちは hello",
        "مرحبا hello",
        "Здравствуй hello",
        "Hola, bonjour, guten Tag!"
    ]
    
    for text in test_cases:
        result = detector.detect_language(text)
        detected = ', '.join(result.detected_languages) if result.detected_languages else "None"
        print(f"'{text:20}' → {detected:20} ({result.confidence:.1%})")


def security_and_privacy_example():
    """Demonstrate security and privacy features."""
    print("\n🔒 Security & Privacy Features")
    print("-" * 40)
    
    # Initialize security components
    privacy_protector = PrivacyProtector()
    security_monitor = SecurityMonitor()
    input_validator = InputValidator()
    detector = FastTextDetector()
    
    text = "Hello, my email is john@example.com and I live at 123 Main St."
    
    # Validate input
    validation = input_validator.validate(text)
    print(f"Input validation: {'✅ Valid' if validation.is_valid else '❌ Invalid'}")
    
    if validation.is_valid:
        # Apply privacy protection
        privacy_result = privacy_protector.protect_text(validation.sanitized_text)
        print(f"PII detected: {len(privacy_result['pii_detected'])} items")
        print(f"Protected text: '{privacy_result['protected_text'][:50]}...'")
        
        # Perform detection on protected text
        result = detector.detect_language(privacy_result['protected_text'])
        print(f"Detection result: {', '.join(result.detected_languages)} ({result.confidence:.1%})")
        
        # Monitor security events
        events = security_monitor.process_request(
            source_id="example_request",
            request_data={'text_length': len(text)},
            user_id="demo_user"
        )
        print(f"Security events: {len(events)} logged")


if __name__ == "__main__":
    print("🌍 SwitchPrint Library v2.1.0 - Basic Examples")
    print("=" * 60)
    
    try:
        basic_detection_example()
        user_guided_detection()
        threshold_mode_analysis()
        multilingual_detection()
        security_and_privacy_example()
        
        print("\n🚀 Next Steps:")
        print("- Try the CLI: python cli.py")
        print("- Install from PyPI: pip install switchprint")
        print("- See examples/advanced_usage.py for more features")
        print("- Check README.md for complete documentation")
        print("- Run tests: pytest tests/test_comprehensive_suite.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install required dependencies: pip install switchprint[all]")
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("💡 Check that all components are properly installed")