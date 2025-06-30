#!/usr/bin/env python3
"""
Basic usage examples for the Code-Switch Aware AI Library
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codeswitch_ai import OptimizedCodeSwitchDetector


def basic_detection_example():
    """Basic language detection example."""
    print("🔍 Basic Language Detection")
    print("-" * 40)
    
    detector = OptimizedCodeSwitchDetector()
    
    # Simple detection
    result = detector.analyze_optimized("Hello, ¿cómo estás?")
    
    print(f"Text: 'Hello, ¿cómo estás?'")
    print(f"Detected languages: {result.detected_languages}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Switch points: {result.switch_points}")


def user_guided_detection():
    """User-guided detection for better accuracy."""
    print("\n👤 User-Guided Detection")
    print("-" * 40)
    
    detector = OptimizedCodeSwitchDetector()
    
    # With user language guidance
    result = detector.analyze_optimized(
        "Main ghar ja raha hoon but I will come back",
        user_languages=["hindi", "english"]
    )
    
    print(f"Text: 'Main ghar ja raha hoon but I will come back'")
    print(f"User languages: Hindi, English")
    print(f"Detected languages: {result.detected_languages}")
    print(f"User language match: {result.user_language_match}")
    print(f"Romanization detected: {result.romanization_detected}")


def phrase_analysis():
    """Analyze phrase-level language clustering."""
    print("\n📋 Phrase-Level Analysis")
    print("-" * 40)
    
    detector = OptimizedCodeSwitchDetector()
    
    result = detector.analyze_optimized(
        "Je suis tired today. Need some café.",
        user_languages=["french", "english"]
    )
    
    print(f"Text: 'Je suis tired today. Need some café.'")
    print("Phrase breakdown:")
    for i, phrase in enumerate(result.phrases, 1):
        marker = "👤" if phrase.get('is_user_language', False) else "🌐"
        print(f"  {i}. {marker} '{phrase['text']}' → {phrase['language']} ({phrase['confidence']:.1%})")


def native_script_detection():
    """Demonstrate native script detection."""
    print("\n📜 Native Script Detection")
    print("-" * 40)
    
    detector = OptimizedCodeSwitchDetector()
    
    test_cases = [
        ("你好 hello world", ["chinese", "english"]),
        ("こんにちは hello", ["japanese", "english"]),
        ("مرحبا hello", ["arabic", "english"]),
        ("Здравствуй hello", ["russian", "english"]),
    ]
    
    for text, user_langs in test_cases:
        result = detector.analyze_optimized(text, user_langs)
        print(f"'{text}' → {result.detected_languages} (native script: {result.native_script_detected})")


def underserved_language_support():
    """Showcase support for underserved languages."""
    print("\n🌍 Underserved Language Support")
    print("-" * 40)
    
    detector = OptimizedCodeSwitchDetector()
    
    examples = [
        ("Saya berbahasa Indonesia", ["indonesian"], "Indonesian"),
        ("Naan Tamil pesuren", ["tamil"], "Tamil"),
        ("Hun Gujarati chhu", ["gujarati"], "Gujarati"),
        ("Ana natakallam al arabiya", ["arabic"], "Arabic (romanized)"),
        ("Habari gani rafiki", ["swahili"], "Swahili"),
    ]
    
    for text, user_langs, description in examples:
        result = detector.analyze_optimized(text, user_langs)
        status = "✅" if result.detected_languages else "❌"
        detected = ', '.join(result.detected_languages) if result.detected_languages else "None"
        print(f"{status} {description:20} | '{text}' → {detected}")


if __name__ == "__main__":
    print("🌍 Code-Switch Aware AI Library - Basic Examples")
    print("=" * 60)
    
    basic_detection_example()
    user_guided_detection()
    phrase_analysis()
    native_script_detection()
    underserved_language_support()
    
    print("\n🚀 Next Steps:")
    print("- Try the CLI: python cli.py")
    print("- See examples/advanced_usage.py for more features")
    print("- Check README.md for complete documentation")