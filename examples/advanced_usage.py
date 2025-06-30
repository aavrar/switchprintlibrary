#!/usr/bin/env python3
"""
Advanced usage examples for the Code-Switch Aware AI Library
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codeswitch_ai import (
    OptimizedCodeSwitchDetector, 
    ConversationMemory, 
    EmbeddingGenerator,
    SimilarityRetriever
)


def conversation_memory_example():
    """Demonstrate conversation memory and retrieval."""
    print("💾 Conversation Memory Example")
    print("-" * 40)
    
    # Initialize components
    detector = OptimizedCodeSwitchDetector()
    memory = ConversationMemory(db_path="demo_conversations.db")
    embedder = EmbeddingGenerator()
    retriever = SimilarityRetriever(memory)
    
    # Store some conversations
    conversations = [
        "Hello, ¿cómo estás? I'm doing bien today.",
        "Je suis tired but need to trabajo.",
        "Main office ja raha hoon, see you later.",
        "Mixing languages comes naturally cuando hablas múltiples idiomas."
    ]
    
    print("Storing conversations:")
    for i, text in enumerate(conversations, 1):
        # Analyze text
        result = detector.analyze_optimized(text, ["english", "spanish", "french", "hindi"])
        
        # Generate embeddings
        stats = {
            'total_switches': len(result.switch_points),
            'detected_languages': result.detected_languages,
            'confidence': result.confidence,
            'romanization_detected': result.romanization_detected
        }
        
        embeddings = embedder.generate_conversation_embedding({
            'text': text,
            'switch_stats': stats,
            'metadata': {'user_id': 'demo_user', 'session_id': f'session_{i}'}
        })
        
        # Store in memory
        from codeswitch_ai.memory.conversation_memory import ConversationEntry
        entry = ConversationEntry(
            text=text,
            switch_stats=stats,
            embeddings=embeddings,
            user_id='demo_user',
            session_id=f'session_{i}'
        )
        
        entry_id = memory.store_conversation(entry)
        print(f"  {i}. Stored: '{text[:40]}...' (ID: {entry_id})")
    
    # Search for similar conversations
    print(f"\nSearching for conversations similar to 'mixing languages':")
    retriever.build_index(user_id='demo_user')
    
    try:
        similar = retriever.search_by_text(
            "mixing languages", embedder, user_id='demo_user', k=3
        )
        
        for i, (conv, score) in enumerate(similar, 1):
            print(f"  {i}. Score: {score:.3f} - '{conv.text[:50]}...'")
    except Exception as e:
        print(f"  Search not available: {e}")
    
    # Cleanup
    import os
    if os.path.exists("demo_conversations.db"):
        os.remove("demo_conversations.db")


def performance_benchmarking():
    """Benchmark performance across different text types."""
    print("\n⚡ Performance Benchmarking")
    print("-" * 40)
    
    detector = OptimizedCodeSwitchDetector()
    
    test_cases = [
        ("Short text", "Hello world"),
        ("Medium text", "Hello, ¿cómo estás? Je suis tired today."),
        ("Long text", " ".join(["Hello, ¿cómo estás? Je suis tired."] * 5)),
        ("Native script", "こんにちは hello 你好 world مرحبا"),
        ("Romanized", "Main ghar ja raha hoon lekin aap kaise hain?"),
    ]
    
    iterations = 50
    
    for description, text in test_cases:
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            result = detector.analyze_optimized(text, ["english", "spanish", "french"])
        avg_time = (time.time() - start) / iterations
        
        print(f"{description:15} | {len(text.split()):2d} words | {avg_time*1000:5.1f}ms | {len(result.detected_languages)} langs")


def multilingual_comparison():
    """Compare detection across different language families."""
    print("\n🌍 Multilingual Language Family Comparison")
    print("-" * 50)
    
    detector = OptimizedCodeSwitchDetector()
    
    test_cases = [
        # European languages
        ("Romance", "Hola, bonjour, ciao, hello", ["spanish", "french", "italian", "english"]),
        ("Germanic", "Hallo, hello, hej, guten Tag", ["german", "english", "swedish", "german"]),
        ("Slavic", "Привет, cześć, zdravo, hello", ["russian", "polish", "serbian", "english"]),
        
        # Asian languages
        ("East Asian", "你好, こんにちは, 안녕하세요, hello", ["chinese", "japanese", "korean", "english"]),
        ("South Asian", "नमस्ते, سلام, hello", ["hindi", "urdu", "english"]),
        ("Southeast Asian", "สวัสดี, xin chào, hello", ["thai", "vietnamese", "english"]),
        
        # African languages
        ("African", "Habari, sawubona, hello", ["swahili", "zulu", "english"]),
        
        # Romanized mixing
        ("Romanized", "Main aap ko hello kehta hoon", ["hindi", "english"]),
    ]
    
    for family, text, user_langs in test_cases:
        result = detector.analyze_optimized(text, user_langs)
        detected = ', '.join(result.detected_languages) if result.detected_languages else "None"
        native_script = "📜" if result.native_script_detected else "  "
        romanization = "🔤" if result.romanization_detected else "  "
        
        print(f"{family:12} {native_script}{romanization} | {detected:20} | {result.confidence:.1%}")


def error_handling_and_edge_cases():
    """Test error handling and edge cases."""
    print("\n🔧 Error Handling & Edge Cases")
    print("-" * 40)
    
    detector = OptimizedCodeSwitchDetector()
    
    edge_cases = [
        ("Empty string", ""),
        ("Whitespace only", "   "),
        ("Numbers only", "123 456 789"),
        ("Punctuation", "!@#$%^&*()"),
        ("Mixed symbols", "Hello! 123 ¿Cómo? 🎉"),
        ("Very long word", "Supercalifragilisticexpialidocious"),
        ("Single character", "a"),
        ("Emojis", "😀🌍🚀💻"),
    ]
    
    for description, text in edge_cases:
        try:
            result = detector.analyze_optimized(text)
            status = "✅"
            info = f"{len(result.detected_languages)} langs, {result.confidence:.1%}"
        except Exception as e:
            status = "❌"
            info = f"Error: {str(e)[:30]}..."
        
        print(f"{status} {description:15} | {info}")


def language_confidence_analysis():
    """Analyze confidence scores across different scenarios."""
    print("\n📊 Language Confidence Analysis")
    print("-" * 40)
    
    detector = OptimizedCodeSwitchDetector()
    
    scenarios = [
        ("Clear English", "The quick brown fox jumps over the lazy dog", ["english"]),
        ("Clear Spanish", "El gato está durmiendo en la casa", ["spanish"]),
        ("Mixed high-conf", "Hello, ¿cómo estás?", ["english", "spanish"]),
        ("Mixed medium", "Je suis tired today", ["french", "english"]),
        ("Romanized unclear", "kya hal hai bhai", ["hindi"]),
        ("Very short", "Hi", ["english"]),
        ("Function words", "the el la and y", ["english", "spanish"]),
    ]
    
    for description, text, user_langs in scenarios:
        result = detector.analyze_optimized(text, user_langs)
        
        confidence_level = "🟢" if result.confidence > 0.8 else "🟡" if result.confidence > 0.5 else "🔴"
        
        print(f"{confidence_level} {description:15} | {result.confidence:.1%} | {', '.join(result.detected_languages)}")


if __name__ == "__main__":
    print("🚀 Code-Switch Aware AI Library - Advanced Examples")
    print("=" * 60)
    
    conversation_memory_example()
    performance_benchmarking()
    multilingual_comparison()
    error_handling_and_edge_cases()
    language_confidence_analysis()
    
    print(f"\n✨ Advanced Features Demonstrated:")
    print("- Conversation memory and similarity search")
    print("- Performance benchmarking across text types")
    print("- Multilingual support across language families")
    print("- Robust error handling for edge cases")
    print("- Confidence analysis for different scenarios")