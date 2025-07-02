#!/usr/bin/env python3
"""
Create comprehensive evaluation suite with synthetic multilingual data.
Alternative to LinCE benchmark with known ground truth.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_synthetic_test_data():
    """Create test data with known ground truth for evaluation."""
    
    # Basic single-language tests (should be 100% accurate)
    basic_tests = [
        # English
        ("Hello world, how are you today?", ["en"], 0),
        ("The quick brown fox jumps over the lazy dog.", ["en"], 0),
        ("I love programming and artificial intelligence.", ["en"], 0),
        
        # Spanish
        ("Hola mundo, ¿cómo estás hoy?", ["es"], 0),
        ("El rápido zorro marrón salta sobre el perro perezoso.", ["es"], 0),
        ("Me encanta la programación y la inteligencia artificial.", ["es"], 0),
        
        # French  
        ("Bonjour le monde, comment allez-vous aujourd'hui?", ["fr"], 0),
        ("Le renard brun rapide saute par-dessus le chien paresseux.", ["fr"], 0),
        ("J'aime la programmation et l'intelligence artificielle.", ["fr"], 0),
        
        # German
        ("Hallo Welt, wie geht es dir heute?", ["de"], 0),
        ("Der schnelle braune Fuchs springt über den faulen Hund.", ["de"], 0),
        ("Ich liebe Programmierung und künstliche Intelligenz.", ["de"], 0),
        
        # Italian
        ("Ciao mondo, come stai oggi?", ["it"], 0),
        ("La volpe marrone veloce salta sopra il cane pigro.", ["it"], 0),
        ("Amo la programmazione e l'intelligenza artificiale.", ["it"], 0),
    ]
    
    # Simple code-switching tests (known switch points)
    switch_tests = [
        # English-Spanish
        ("Hello, ¿cómo estás? I'm doing well today.", ["en", "es"], 2),
        ("I want to go to the playa tomorrow.", ["en", "es"], 1),
        ("Vamos to the store right now.", ["es", "en"], 1),
        
        # English-French
        ("Hello, comment allez-vous? I'm fine merci.", ["en", "fr"], 2), 
        ("I love this café, it's très good.", ["en", "fr"], 1),
        ("Bonjour my friend, how are you?", ["fr", "en"], 1),
        
        # Spanish-French
        ("Hola, comment ça va? Muy bien merci.", ["es", "fr"], 2),
        ("Me gusta cette musique beaucoup.", ["es", "fr"], 1),
        
        # English-German
        ("Hello, wie geht es dir? I'm gut today.", ["en", "de"], 2),
        ("That's sehr interesting, don't you think?", ["en", "de"], 1),
        
        # Multi-language
        ("Hello, bonjour, hola everyone!", ["en", "fr", "es"], 2),
        ("I love cette película, it's muy good.", ["en", "fr", "es"], 2),
    ]
    
    # Social media style tests
    social_media_tests = [
        ("omg this concert was amazing! que buena música 🎵", ["en", "es"], 1),
        ("Going to work pero I'm so tired today", ["en", "es"], 1),
        ("je suis so excited for the weekend!", ["fr", "en"], 1),
        ("Actually nunca mind, let's stay home", ["en", "es"], 1),
        ("This película was increíble, highly recommend", ["en", "es"], 1),
    ]
    
    return {
        "basic_single_language": basic_tests,
        "simple_code_switching": switch_tests, 
        "social_media_style": social_media_tests
    }

def evaluate_detector(detector, test_data, category_name):
    """Evaluate detector on test category."""
    print(f"\n=== {category_name} ===")
    
    correct_primary = 0
    correct_multilingual = 0
    total = len(test_data)
    
    for text, expected_langs, expected_switches in test_data:
        try:
            result = detector.detect_language(text)
            
            # Check primary language detection
            if result.detected_languages and result.detected_languages[0] in expected_langs:
                correct_primary += 1
            
            # Check multilingual detection
            detected_set = set(result.detected_languages)
            expected_set = set(expected_langs)
            if len(expected_set & detected_set) >= min(len(expected_set), len(detected_set)):
                correct_multilingual += 1
            
            print(f"'{text[:50]}...' -> {result.detected_languages[:3]} (conf: {result.confidence:.3f})")
            
        except Exception as e:
            print(f"Error on '{text}': {e}")
    
    primary_acc = correct_primary / total * 100
    multilingual_acc = correct_multilingual / total * 100
    
    print(f"\nResults for {category_name}:")
    print(f"Primary Language Accuracy: {primary_acc:.1f}% ({correct_primary}/{total})")
    print(f"Multilingual Detection: {multilingual_acc:.1f}% ({correct_multilingual}/{total})")
    
    return primary_acc, multilingual_acc

def main():
    """Run comprehensive evaluation."""
    print("SwitchPrint Evaluation Suite")
    print("=" * 50)
    
    # Load test data
    test_data = create_synthetic_test_data()
    
    # Test current implementation
    try:
        from codeswitch_ai import FastTextDetector, EnsembleDetector
        
        print("\n🔬 Testing FastText Detector")
        fasttext_detector = FastTextDetector()
        
        ft_results = {}
        for category, data in test_data.items():
            primary_acc, multi_acc = evaluate_detector(fasttext_detector, data, category)
            ft_results[category] = (primary_acc, multi_acc)
        
        print("\n🔬 Testing Ensemble Detector")
        try:
            ensemble_detector = EnsembleDetector(use_fasttext=True, use_transformer=False)
            
            ens_results = {}
            for category, data in test_data.items():
                primary_acc, multi_acc = evaluate_detector(ensemble_detector, data, category)
                ens_results[category] = (primary_acc, multi_acc)
                
        except Exception as e:
            print(f"Ensemble detector failed: {e}")
            ens_results = {}
        
        # Summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        
        for category in test_data.keys():
            print(f"\n{category.upper()}:")
            if category in ft_results:
                ft_primary, ft_multi = ft_results[category]
                print(f"  FastText:  Primary {ft_primary:.1f}%, Multi {ft_multi:.1f}%")
            if category in ens_results:
                ens_primary, ens_multi = ens_results[category]
                print(f"  Ensemble:  Primary {ens_primary:.1f}%, Multi {ens_multi:.1f}%")
        
        # Overall averages
        if ft_results:
            ft_avg_primary = sum(r[0] for r in ft_results.values()) / len(ft_results)
            ft_avg_multi = sum(r[1] for r in ft_results.values()) / len(ft_results)
            print(f"\nFastText Overall:  Primary {ft_avg_primary:.1f}%, Multi {ft_avg_multi:.1f}%")
        
        if ens_results:
            ens_avg_primary = sum(r[0] for r in ens_results.values()) / len(ens_results)
            ens_avg_multi = sum(r[1] for r in ens_results.values()) / len(ens_results)
            print(f"Ensemble Overall:  Primary {ens_avg_primary:.1f}%, Multi {ens_avg_multi:.1f}%")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()