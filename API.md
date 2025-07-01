# API Reference

Complete API documentation for SwitchPrint v2.0.0.

## Core Detection Components

### EnsembleDetector
The main detector combining FastText, Transformer, and rule-based methods.

```python
class EnsembleDetector:
    def __init__(
        self,
        use_fasttext: bool = True,
        use_transformer: bool = True,
        transformer_model: str = "bert-base-multilingual-cased",
        ensemble_strategy: str = "weighted_average",  # "voting", "confidence_based"
        cache_size: int = 1000
    )
```

#### Methods

**detect_language(text, user_languages=None)**
- `text` (str): Input text to analyze
- `user_languages` (List[str], optional): User's known languages for improved accuracy
- Returns: `EnsembleResult` with detected languages, confidence, and metadata

**detect_languages_batch(texts, user_languages=None)**
- `texts` (List[str]): List of texts to analyze
- `user_languages` (List[str], optional): User's known languages
- Returns: List of `EnsembleResult` objects

### FastTextDetector
High-performance language detection using Facebook's FastText.

```python
class FastTextDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        cache_size: int = 1000
    )
```

#### Methods

**detect_language(text, user_languages=None)**
- Fastest detection method (0.1-0.6ms)
- 85.98% accuracy on multilingual text
- Supports 176 languages

### TransformerDetector
Contextual language detection using BERT-based models.

```python
class TransformerDetector:
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        device: str = "auto",  # "cpu", "cuda", "auto"
        max_length: int = 512
    )
```

#### Methods

**detect_language(text, user_languages=None)**
- Contextual understanding with mBERT
- Best for complex code-switching patterns
- Slower but more accurate for ambiguous cases

## Memory and Storage

### ConversationMemory
Persistent storage for multilingual conversations.

```python
class ConversationMemory:
    def __init__(
        self,
        db_path: str = "conversations.db",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    )
```

#### Methods

**store_conversation(user_id, text, metadata=None)**
- `user_id` (str): User identifier
- `text` (str): Conversation text
- `metadata` (dict, optional): Additional metadata
- Returns: Conversation ID

**get_user_conversations(user_id, limit=50)**
- Retrieve conversations for specific user
- Returns: List of `ConversationEntry` objects

**search_conversations(query, user_id=None, limit=10)**
- Semantic search through stored conversations
- Returns: List of relevant conversations with similarity scores

### EmbeddingGenerator
Multilingual text embeddings for similarity search.

```python
class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "auto"
    )
```

## Retrieval and Search

### OptimizedSimilarityRetriever
GPU-accelerated similarity search with FAISS.

```python
class OptimizedSimilarityRetriever:
    def __init__(
        self,
        memory: ConversationMemory,
        use_gpu: bool = True,
        index_type: str = "auto",  # "flat", "ivf", "hnsw", "auto"
        quantization: bool = True
    )
```

#### Methods

**search_similar(query, user_id=None, limit=10)**
- High-performance similarity search
- Sub-millisecond response time
- GPU acceleration when available

**build_index(force_rebuild=False)**
- Build optimized search indices
- Automatic index type selection based on data size

**get_index_statistics()**
- Returns performance metrics and index information

## Security Components

### InputValidator
Input validation and sanitization for secure text processing.

```python
class InputValidator:
    def __init__(self, config: SecurityConfig)
```

#### SecurityConfig
```python
@dataclass
class SecurityConfig:
    security_level: str = "moderate"  # "permissive", "moderate", "strict", "paranoid"
    max_text_length: int = 50000
    enable_html_sanitization: bool = True
    enable_injection_detection: bool = True
    enable_pii_detection: bool = True
    blocked_patterns: List[str] = None
```

#### Methods

**validate(text, source="unknown")**
- Comprehensive input validation
- Returns: `ValidationResult` with sanitized text and threat analysis

### PrivacyProtector
PII detection and data anonymization.

```python
class PrivacyProtector:
    def __init__(self, config: Optional[PrivacyConfig] = None)
```

#### PrivacyConfig
```python
@dataclass
class PrivacyConfig:
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD
    enabled_detectors: Set[PIIType] = None
    anonymization_method: str = "replacement"  # "replacement", "hashing", "masking"
    preserve_language_structure: bool = True
```

#### Privacy Levels
- `PrivacyLevel.MINIMAL`: Basic PII detection
- `PrivacyLevel.STANDARD`: Extended PII detection
- `PrivacyLevel.HIGH`: Comprehensive PII detection
- `PrivacyLevel.MAXIMUM`: Full anonymization with fake data

#### Methods

**protect_text(text, source_id=None)**
- Detect and anonymize PII in text
- Returns: Dictionary with protected text and metadata

**batch_protect(texts, source_ids=None)**
- Process multiple texts efficiently
- Returns: List of protection results

### SecurityMonitor
Real-time security monitoring and threat detection.

```python
class SecurityMonitor:
    def __init__(self, log_file: Optional[str] = None)
```

#### Methods

**process_request(source_id, request_data, user_id=None, ip_address=None, success=True)**
- Monitor requests for security threats
- Returns: List of detected security events

**generate_security_report(hours=24)**
- Generate comprehensive security report
- Returns: Security analysis and statistics

### ModelSecurityAuditor
ML model security auditing and integrity checking.

```python
class ModelSecurityAuditor:
    def __init__(self, trusted_sources: Optional[List[str]] = None)
```

#### Methods

**audit_model_file(file_path, expected_hash=None)**
- Comprehensive security audit of model files
- Returns: `SecurityScanResult` with threat assessment

**generate_security_report(results=None)**
- Aggregate security report across multiple models
- Returns: Security summary and recommendations

## Streaming and Real-time Processing

### StreamingDetector
Real-time code-switching detection for live conversations.

```python
class StreamingDetector:
    def __init__(
        self,
        detector: LanguageDetector,
        config: StreamingConfig = None
    )
```

#### StreamingConfig
```python
@dataclass
class StreamingConfig:
    chunk_size: int = 50
    overlap_size: int = 10
    buffer_size: int = 1000
    min_confidence: float = 0.7
    enable_context_carryover: bool = True
```

#### Methods

**process_chunk(text, timestamp=None)**
- Process text chunk in streaming fashion
- Returns: `StreamResult` with real-time analysis

**get_conversation_state()**
- Get current conversation state and statistics
- Returns: `ConversationState` object

### RealTimeAnalyzer
Advanced real-time conversation analysis.

```python
class RealTimeAnalyzer:
    def __init__(
        self,
        detector: StreamingDetector,
        memory: ConversationMemory = None
    )
```

## Evaluation and Benchmarking

### LinCEBenchmark
Integration with LinCE research benchmark.

```python
class LinCEBenchmark:
    def __init__(self, dataset_path: str = None)
```

#### Methods

**evaluate_detector(detector, dataset="all")**
- Evaluate detector against LinCE benchmark
- Returns: Detailed performance metrics

### MTEBEvaluator
Massive Text Embedding Benchmark evaluation.

```python
class MTEBEvaluator:
    def __init__(self, tasks: List[str] = None)
```

#### Methods

**evaluate_embeddings(embedding_model, tasks=None)**
- Evaluate embeddings on MTEB tasks
- Returns: Comprehensive evaluation results

## Training Components

### FastTextDomainTrainer
Custom FastText model training for domain-specific data.

```python
class FastTextDomainTrainer:
    def __init__(self, config: FineTuningConfig)
```

#### Methods

**train_domain_model(training_data, output_path)**
- Train custom FastText model
- Returns: Training statistics and model path

**create_synthetic_data(languages, samples_per_lang=1000)**
- Generate synthetic training data
- Returns: Synthetic dataset for training

## Advanced Features

### TemporalCodeSwitchAnalyzer
Analyze code-switching patterns over time.

```python
class TemporalCodeSwitchAnalyzer:
    def __init__(self, memory: ConversationMemory)
```

#### Methods

**analyze_user_patterns(user_id, time_range_days=30)**
- Analyze temporal switching patterns
- Returns: `TemporalStatistics` with pattern analysis

### ContextAwareClusterer
Advanced phrase clustering using mBERT.

```python
class ContextAwareClusterer:
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        clustering_threshold: float = 0.8
    )
```

## Data Classes and Results

### EnsembleResult
```python
@dataclass
class EnsembleResult:
    detected_languages: List[str]
    confidence: float
    ensemble_weights: Dict[str, float]
    method_results: Dict[str, Any]
    switch_points: List[Tuple[int, str, str]]
    phrases: List[Dict[str, Any]]
    processing_time: float
```

### ValidationResult
```python
@dataclass
class ValidationResult:
    is_valid: bool
    sanitized_text: str
    threats_detected: List[str]
    security_score: float
    warnings: List[str]
    original_length: int
    sanitized_length: int
```

### SecurityScanResult
```python
@dataclass
class SecurityScanResult:
    model_path: str
    is_safe: bool
    threat_level: SecurityThreatLevel
    issues_detected: List[ModelSecurityIssue]
    warnings: List[str]
    file_hash: str
    file_size: int
    scan_timestamp: float
    recommendations: List[str]
```

## Error Handling

### Common Exceptions
- `DetectionError`: Raised when detection fails
- `SecurityViolationError`: Raised for security policy violations
- `ModelLoadError`: Raised when model loading fails
- `ValidationError`: Raised for input validation failures

### Error Handling Example
```python
from codeswitch_ai import EnsembleDetector, DetectionError

try:
    detector = EnsembleDetector()
    result = detector.detect_language("Hello world")
except DetectionError as e:
    print(f"Detection failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### Memory Usage
- FastText models: ~100MB
- Transformer models: ~500MB-2GB
- FAISS indices: Varies with data size
- Conversation storage: SQLite overhead minimal

### Speed Optimization
- Use FastText for speed-critical applications
- Enable GPU acceleration for large-scale processing
- Use appropriate FAISS index types
- Cache frequently used results

### Batch Processing
All detectors support batch processing for improved throughput:
```python
texts = ["text1", "text2", "text3"]
results = detector.detect_languages_batch(texts)
```

## Configuration Files

### Environment Variables
- `CODESWITCH_MODEL_PATH`: Default model path
- `CODESWITCH_CACHE_SIZE`: Default cache size
- `CODESWITCH_DB_PATH`: Default database path
- `CODESWITCH_LOG_LEVEL`: Logging level

### Configuration File Support
```python
import json
from codeswitch_ai import EnsembleDetector

# Load from config file
with open('config.json', 'r') as f:
    config = json.load(f)

detector = EnsembleDetector(**config['detector_settings'])
```