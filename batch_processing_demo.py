#!/usr/bin/env python3
"""
High-Performance Batch Processing Demo

Demonstrates the optimized batch processing capabilities for high-throughput
code-switching detection applications.
"""

import time
import json
from typing import List
from codeswitch_ai import (
    HighPerformanceBatchProcessor, 
    BatchConfig, 
    IntegratedImprovedDetector
)


def create_demo_dataset() -> List[str]:
    """Create a diverse dataset for demonstration."""
    
    base_texts = [
        # English
        "Hello world, how are you today?",
        "This is a simple English sentence.",
        "The weather is beautiful outside.",
        
        # Spanish  
        "Hola mundo, ¿cómo estás hoy?",
        "Buenos días, que tengas un buen día.",
        "El clima está hermoso afuera.",
        
        # Code-switching (English-Spanish)
        "Hello, ¿cómo estás? I'm doing bien today.",
        "Let's go to the mercado and buy some food.",
        "I love this canción, it's muy bonita.",
        
        # Hindi-English (romanized)
        "I need chai right now, very urgent hai.",
        "Yaar, this movie is amazing, dekho.",
        "Jaldi bro, we need to leave abhi.",
        
        # Arabic-English  
        "Yallah chalein, let's go now.",
        "Habibi, how are you feeling today?",
        "Inshallah everything will be fine.",
        
        # French-English
        "Je suis très tired aujourd'hui, tu sais?",
        "This café has the best croissants ever.",
        "Bonjour! How are you doing today?",
        
        # Multi-language
        "Hello, je suis going to the mercado today.",
        "This película is tres interesting, no?",
        "I think we should aller to the playa.",
        
        # Short texts
        "OK",
        "Gracias",
        "Merci beaucoup",
        "Shukriya bhai",
        
        # Longer texts
        "This is a longer English sentence that contains multiple ideas and should be processed efficiently by the batch system.",
        "Este es un texto más largo en español que contiene múltiples ideas y debería ser procesado de manera eficiente.",
        "This longer text switches between English and español to test the detector's ability to handle mixed content efficiently."
    ]
    
    # Multiply to create realistic batch size
    return base_texts * 50  # 1,350 texts total


def run_performance_comparison():
    """Compare different batch processing configurations."""
    
    print("🏁 BATCH PROCESSING PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Create test dataset
    texts = create_demo_dataset()
    print(f"📝 Dataset: {len(texts):,} texts")
    print(f"   Unique texts: {len(set(texts))}")
    print(f"   Average length: {sum(len(t.split()) for t in texts) / len(texts):.1f} words")
    
    # Test configurations
    configs = [
        ("Conservative", BatchConfig(
            max_workers=1,
            chunk_size=50,
            enable_caching=False
        )),
        ("Balanced", BatchConfig(
            max_workers=4,
            chunk_size=100,
            enable_caching=True
        )),
        ("High-Performance", BatchConfig(
            max_workers=8,
            chunk_size=200,
            enable_caching=True,
            cache_size=20000
        )),
        ("Ultra-Fast", BatchConfig(
            max_workers=16,
            chunk_size=500,
            enable_caching=True,
            cache_size=50000
        ))
    ]
    
    results = []
    
    for config_name, config in configs:
        print(f"\n🔧 Testing {config_name} Configuration")
        print(f"   Workers: {config.max_workers}, Chunk: {config.chunk_size}, Cache: {config.enable_caching}")
        
        # Create processor
        detector = IntegratedImprovedDetector(
            performance_mode="fast",
            auto_train_calibration=False
        )
        processor = HighPerformanceBatchProcessor(detector=detector, config=config)
        
        # Run benchmark
        start_time = time.time()
        result = processor.process_batch(texts)
        end_time = time.time()
        
        # Collect metrics
        metrics = {
            'config_name': config_name,
            'total_time': end_time - start_time,
            'texts_per_second': result.metrics.texts_per_second,
            'cache_hit_rate': result.metrics.cache_hit_rate,
            'memory_usage_mb': result.metrics.memory_usage_mb,
            'worker_efficiency': result.metrics.worker_efficiency,
            'processed_texts': result.metrics.processed_texts,
            'failed_texts': result.metrics.failed_texts
        }
        
        results.append(metrics)
        
        print(f"   ⚡ Speed: {metrics['texts_per_second']:,.0f} texts/sec")
        print(f"   💾 Cache hits: {metrics['cache_hit_rate']:.1%}")
        print(f"   🧠 Memory: {metrics['memory_usage_mb']:.0f} MB")
        print(f"   ✅ Success: {metrics['processed_texts']}/{len(texts)}")
    
    # Summary
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    best_speed = max(results, key=lambda x: x['texts_per_second'])
    best_cache = max(results, key=lambda x: x['cache_hit_rate'])
    best_memory = min(results, key=lambda x: x['memory_usage_mb'])
    
    print(f"🚀 Fastest: {best_speed['config_name']} ({best_speed['texts_per_second']:,.0f} texts/sec)")
    print(f"🎯 Best Cache: {best_cache['config_name']} ({best_cache['cache_hit_rate']:.1%} hit rate)")
    print(f"💾 Most Efficient: {best_memory['config_name']} ({best_memory['memory_usage_mb']:.0f} MB)")
    
    return results


def demonstrate_streaming_processing():
    """Demonstrate streaming text processing."""
    
    print("\n🌊 STREAMING PROCESSING DEMO")
    print("=" * 50)
    
    def text_generator():
        """Generate streaming text data."""
        base_texts = [
            "Hello world",
            "Hola mundo", 
            "I need chai now",
            "Yallah chalein",
            "Bonjour monde",
            "Guten Tag world"
        ]
        
        for i in range(100):  # Stream 100 texts
            for text in base_texts:
                yield f"{text} (stream #{i+1})"
    
    # Create processor
    config = BatchConfig(
        max_workers=4,
        chunk_size=20,
        enable_caching=True
    )
    
    detector = IntegratedImprovedDetector(
        performance_mode="fast",
        auto_train_calibration=False
    )
    processor = HighPerformanceBatchProcessor(detector=detector, config=config)
    
    # Collect results
    streamed_results = []
    
    def result_callback(result):
        streamed_results.append(result)
        if len(streamed_results) % 50 == 0:
            print(f"   📥 Processed {len(streamed_results)} texts...")
    
    # Process stream
    start_time = time.time()
    metrics = processor.process_stream(
        text_generator(),
        result_callback,
        buffer_size=30
    )
    end_time = time.time()
    
    print(f"✅ Streaming complete!")
    print(f"   📊 Processed: {metrics.processed_texts} texts")
    print(f"   ⚡ Speed: {metrics.texts_per_second:.0f} texts/sec")
    print(f"   ⏱️ Total time: {end_time - start_time:.2f}s")
    print(f"   🎯 Cache hit rate: {metrics.cache_hit_rate:.1%}")


def demonstrate_real_world_use_cases():
    """Demonstrate real-world batch processing scenarios."""
    
    print("\n🌍 REAL-WORLD USE CASES")
    print("=" * 50)
    
    # Use case 1: Social media analysis
    print("\n📱 Use Case 1: Social Media Post Analysis")
    social_media_posts = [
        "Just finished my homework! Time to chill 😊",
        "Omg this concert was amazing!! Best night ever 🎵",
        "Coffee time ☕ Need my caffeine fix right now",
        "Study session with amigos tonight 📚",
        "Que buen día! Perfect weather for the beach 🏖️",
        "Missing home food 😢 Nothing beats mama's cooking",
        "Work meeting in 5 mins, wish me luck! 💼",
        "Weekend plans: Netflix and chill with bae 💕"
    ] * 20  # 160 posts
    
    config = BatchConfig(
        max_workers=6,
        chunk_size=40,
        enable_caching=True
    )
    
    detector = IntegratedImprovedDetector(
        performance_mode="balanced",
        detector_mode="code_switching",
        auto_train_calibration=False
    )
    processor = HighPerformanceBatchProcessor(detector=detector, config=config)
    
    result = processor.process_batch(social_media_posts)
    
    # Analyze results
    code_mixed_posts = [r for r in result.results if r.is_code_mixed]
    
    print(f"   📊 Analyzed: {len(result.results)} social media posts")
    print(f"   🔄 Code-mixed: {len(code_mixed_posts)} ({len(code_mixed_posts)/len(result.results):.1%})")
    print(f"   ⚡ Processing speed: {result.metrics.texts_per_second:.0f} posts/sec")
    
    # Use case 2: Customer support analysis
    print("\n🎧 Use Case 2: Customer Support Ticket Analysis") 
    support_tickets = [
        "My order hasn't arrived yet, can you help?",
        "The product is damaged, necesito un replacement",
        "Billing issue - I was charged twice this month",
        "Great service! Muchas gracias for the quick response",
        "Login problem - can't access my compte",
        "Return request for item purchased last week"
    ] * 30  # 180 tickets
    
    result = processor.process_batch(support_tickets)
    
    multilingual_tickets = [r for r in result.results if len(r.detected_languages) > 1]
    
    print(f"   📊 Analyzed: {len(result.results)} support tickets")
    print(f"   🌍 Multilingual: {len(multilingual_tickets)} ({len(multilingual_tickets)/len(result.results):.1%})")
    print(f"   ⚡ Processing speed: {result.metrics.texts_per_second:.0f} tickets/sec")


def main():
    """Run comprehensive batch processing demonstration."""
    
    print("🚀 HIGH-PERFORMANCE BATCH PROCESSING DEMO")
    print("=" * 70)
    print("Showcasing optimized batch processing for code-switching detection")
    print()
    
    try:
        # Performance comparison
        performance_results = run_performance_comparison()
        
        # Streaming demo
        demonstrate_streaming_processing()
        
        # Real-world use cases
        demonstrate_real_world_use_cases()
        
        print("\n🎉 DEMO COMPLETE!")
        print("=" * 50)
        print("✅ All batch processing features demonstrated successfully")
        print("📈 Performance optimizations working correctly")
        print("🚀 Ready for high-throughput production applications")
        
        # Save results
        with open('batch_processing_results.json', 'w') as f:
            json.dump(performance_results, f, indent=2)
        print("💾 Performance results saved to 'batch_processing_results.json'")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()