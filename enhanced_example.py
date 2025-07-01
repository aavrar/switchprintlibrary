#!/usr/bin/env python3
"""Enhanced example showcasing the improved code-switch aware AI library."""

import os
import sys
import time
from typing import Dict, List

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codeswitch_ai import (
    FastTextDetector, 
    TransformerDetector, 
    EnsembleDetector,
    ConversationMemory, 
    EmbeddingGenerator,
    OptimizedSimilarityRetriever
)
from codeswitch_ai.retrieval.optimized_retriever import SearchResult


def demonstrate_enhanced_detection():
    """Demonstrate the enhanced detection capabilities."""
    print("🔍 Enhanced Language Detection Capabilities")
    print("=" * 50)
    
    # Test texts with various code-switching patterns
    test_texts = [
        "Hello, how are you? ¿Cómo estás? I'm doing bien.",
        "Main ghar ja raha hoon, but I'll be back soon.",
        "Je suis très tired aujourd'hui, tu sais?",
        "Aap kaise hain? I hope you're doing well.",
        "Bonjour! Comment allez-vous? I hope todo está bien.",
        "这个很好 but I think we need more tiempo para finish.",
        "Привет! How are you doing сегодня?",
        "¡Hola amigo! Ready for the meeting? Vamos a discuss todo."
    ]
    
    user_languages = ["english", "spanish", "hindi", "french"]
    
    # Initialize detectors
    print("\n📊 Initializing detectors...")
    
    try:
        fasttext_detector = FastTextDetector()
        print("✓ FastText detector loaded")
    except Exception as e:
        print(f"⚠ FastText detector failed: {e}")
        fasttext_detector = None
    
    try:
        transformer_detector = TransformerDetector(model_name="bert-base-multilingual-cased")
        print("✓ Transformer detector (mBERT) loaded")
    except Exception as e:
        print(f"⚠ Transformer detector failed: {e}")
        transformer_detector = None
    
    try:
        ensemble_detector = EnsembleDetector(
            use_fasttext=fasttext_detector is not None,
            use_transformer=transformer_detector is not None,
            ensemble_strategy="weighted_average"
        )
        print("✓ Ensemble detector loaded")
    except Exception as e:
        print(f"⚠ Ensemble detector failed: {e}")
        ensemble_detector = None
    
    # Compare detection results
    print("\n🔬 Detection Comparison Results:")
    print("-" * 80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: \"{text}\"")
        print("   User languages:", user_languages)
        
        # FastText results
        if fasttext_detector:
            start_time = time.time()
            ft_result = fasttext_detector.detect_language(text, user_languages)
            ft_time = time.time() - start_time
            
            print(f"   FastText:    {ft_result.detected_languages} "
                  f"(conf: {ft_result.confidence:.3f}, time: {ft_time*1000:.1f}ms)")
        
        # Transformer results
        if transformer_detector:
            start_time = time.time()
            tr_result = transformer_detector.detect_language(text, user_languages)
            tr_time = time.time() - start_time
            
            print(f"   Transformer: {tr_result.detected_languages} "
                  f"(conf: {tr_result.confidence:.3f}, time: {tr_time*1000:.1f}ms)")
        
        # Ensemble results
        if ensemble_detector:
            start_time = time.time()
            en_result = ensemble_detector.detect_language(text, user_languages)
            en_time = time.time() - start_time
            
            print(f"   Ensemble:    {en_result.detected_languages} "
                  f"(conf: {en_result.confidence:.3f}, time: {en_time*1000:.1f}ms)")
            
            # Show ensemble details
            if hasattr(en_result, 'ensemble_weights'):
                weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in en_result.ensemble_weights.items()])
                print(f"                Weights: {weights_str}")
            
            # Show switch points if detected
            if hasattr(en_result, 'switch_points') and en_result.switch_points:
                print(f"                Switch points: {len(en_result.switch_points)} detected")
        
        print()


def demonstrate_memory_and_retrieval():
    """Demonstrate the enhanced memory and retrieval system."""
    print("\n💾 Enhanced Memory and Retrieval System")
    print("=" * 50)
    
    # Initialize components
    memory = ConversationMemory()
    embedding_generator = EmbeddingGenerator()  # Uses multilingual model
    retriever = OptimizedSimilarityRetriever(memory, use_gpu=True, index_type="auto")
    
    # Sample conversations for testing
    sample_conversations = [
        {
            "text": "Hello! I'm excited about learning new languages. ¡Muy emocionante!",
            "user_id": "user1",
            "languages": ["english", "spanish"]
        },
        {
            "text": "Main aaj bahut khush hoon. Today was a great day!",
            "user_id": "user1", 
            "languages": ["hindi", "english"]
        },
        {
            "text": "Je suis tired aujourd'hui. Need some rest.",
            "user_id": "user2",
            "languages": ["french", "english"]
        },
        {
            "text": "¿Cómo estás? I hope you're doing bien today.",
            "user_id": "user1",
            "languages": ["spanish", "english"]
        },
        {
            "text": "Aap kaise hain? How are things going with work?",
            "user_id": "user2",
            "languages": ["hindi", "english"]
        }
    ]
    
    print("\n📝 Storing conversations...")
    
    # Store conversations with enhanced embeddings
    for i, conv_data in enumerate(sample_conversations):
        print(f"   Storing conversation {i+1}: \"{conv_data['text'][:50]}...\"")
        
        # Create mock switch stats
        switch_stats = {
            "total_switches": len(conv_data['languages']) - 1,
            "unique_languages": len(conv_data['languages']),
            "languages": conv_data['languages'],
            "switch_density": 0.1 * len(conv_data['languages']),
            "avg_confidence": 0.85
        }
        
        memory.create_and_store_conversation(
            text=conv_data['text'],
            user_id=conv_data['user_id'],
            switch_stats=switch_stats,
            session_id="demo_session",
            metadata={"timestamp": time.time() + i}
        )
    
    print(f"✓ Stored {len(sample_conversations)} conversations")
    
    # Build optimized indices
    print("\n🔧 Building optimized FAISS indices...")
    retriever.build_index(force_rebuild=True)
    
    # Get index statistics
    stats = retriever.get_index_statistics()
    print(f"✓ Built {stats['total_indices']} optimized indices")
    
    for index_name, index_stats in stats['indices'].items():
        print(f"   {index_name}: {index_stats['n_vectors']} vectors, "
              f"{index_stats['index_type']} type, "
              f"{index_stats.get('memory_usage_mb', 0):.1f}MB")
    
    # Demonstrate enhanced search capabilities
    print("\n🔍 Enhanced Search Demonstrations:")
    print("-" * 40)
    
    query_texts = [
        "I'm feeling very happy today",
        "How are you doing?",
        "Learning languages is fun"
    ]
    
    for query in query_texts:
        print(f"\nQuery: \"{query}\"")
        
        # Semantic search
        results = retriever.search_similar(
            embedding_generator.generate_text_embedding(query),
            embedding_type="semantic",
            k=2
        )
        
        print("   Semantic matches:")
        for j, result in enumerate(results):
            print(f"     {j+1}. \"{result.conversation.text[:60]}...\" "
                  f"(similarity: {result.similarity_score:.3f}, "
                  f"search_time: {result.search_time*1000:.1f}ms)")
    
    # Show performance statistics
    search_perf = stats['search_performance']
    if search_perf['total_searches'] > 0:
        avg_search_time = search_perf['total_search_time'] / search_perf['total_searches']
        print(f"\n📈 Search Performance:")
        print(f"   Total searches: {search_perf['total_searches']}")
        print(f"   Average search time: {avg_search_time*1000:.2f}ms")
        print(f"   Cache hit rate: {search_perf['cache_hits']/(search_perf['cache_hits']+search_perf['cache_misses'])*100:.1f}%")


def demonstrate_advanced_features():
    """Demonstrate advanced features and optimizations."""
    print("\n🚀 Advanced Features and Optimizations")
    print("=" * 50)
    
    # Test ensemble detector capabilities
    try:
        ensemble = EnsembleDetector(
            use_fasttext=True,
            use_transformer=True,
            ensemble_strategy="confidence_based"
        )
        
        test_text = "Hello! ¿Cómo estás? Je vais bien, merci!"
        
        print(f"\n🎯 Advanced Ensemble Analysis:")
        print(f"Text: \"{test_text}\"")
        
        result = ensemble.detect_language(test_text, ["english", "spanish", "french"])
        
        print(f"\nResults:")
        print(f"   Primary language: {result.detected_languages}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Final method: {result.final_method}")
        
        if hasattr(result, 'method_results'):
            print(f"\n   Individual detector results:")
            for method, method_result in result.method_results.items():
                print(f"     {method}: {method_result.detected_languages} "
                      f"(conf: {method_result.confidence:.3f})")
        
        # Show switch points if available
        if hasattr(result, 'switch_points') and result.switch_points:
            print(f"\n   Code-switching points detected:")
            for point in result.switch_points:
                print(f"     Position {point[0]}: {point[1]} → {point[2]} "
                      f"(confidence: {point[3]:.3f})")
        
        # Show phrase clusters
        if hasattr(result, 'phrases') and result.phrases:
            print(f"\n   Phrase clusters:")
            for phrase in result.phrases:
                print(f"     \"{phrase['text']}\" → {phrase['language']} "
                      f"(conf: {phrase['confidence']:.3f})")
        
    except Exception as e:
        print(f"⚠ Advanced ensemble features not available: {e}")
    
    # Demonstrate model information
    print(f"\n📋 System Information:")
    
    try:
        ft_detector = FastTextDetector()
        ft_info = ft_detector.get_model_info()
        print(f"   FastText: {ft_info.get('supported_languages', 'Unknown')} languages supported")
    except:
        pass
    
    try:
        tr_detector = TransformerDetector()
        tr_info = tr_detector.get_model_info()
        print(f"   Transformer: {tr_info.get('model_name', 'Unknown')} "
              f"({tr_info.get('num_parameters', 0):,} parameters)")
    except:
        pass
    
    print(f"   Embedding model: paraphrase-multilingual-MiniLM-L12-v2 (50+ languages)")


def main():
    """Main demonstration function."""
    print("🌟 Enhanced Code-Switch Aware AI Library Demo")
    print("=" * 60)
    print("Showcasing FastText, mBERT, Ensemble Detection, and Optimized Retrieval")
    print()
    
    try:
        # Run demonstrations
        demonstrate_enhanced_detection()
        demonstrate_memory_and_retrieval()
        demonstrate_advanced_features()
        
        print("\n✨ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Try the enhanced CLI: python cli.py")
        print("2. Run comprehensive tests: python -m pytest tests/")
        print("3. Check out the new ensemble detection capabilities")
        print("4. Explore optimized FAISS indices with GPU support")
        
    except KeyboardInterrupt:
        print("\n\n⏹ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This may be due to missing dependencies or model files.")
        print("Please check the requirements and try again.")


if __name__ == "__main__":
    main()