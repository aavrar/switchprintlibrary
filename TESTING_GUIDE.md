# 🧪 Testing Guide - Code-Switch Aware AI Library

## ✅ Installation & Setup

First, install dependencies:
```bash
pip install langdetect sentence-transformers faiss-cpu numpy pandas scikit-learn
```

## 🚀 Testing Methods

### 1. Quick Demo - Run Examples
```bash
python example.py
```
This runs 8 different multilingual examples showing:
- English-Spanish code-switching
- Hindi-English romanized
- Urdu-English romanized  
- French-English mixing
- Complex multilingual text
- Arabic-English romanized
- Function word detection

### 2. Simple Functionality Test
```bash
python test_simple.py
```
Tests basic detection with 4 test cases.

### 3. Interactive CLI Testing
```bash
python cli.py
```

#### CLI Commands to Try:
```bash
# Set your languages for better accuracy
set-languages english,spanish

# Test enhanced analysis
enhanced Hello, how are you? ¿Cómo estás?

# Test romanized languages  
enhanced Main school ja raha hoon but I will come back

# Test memory features
remember I love mixing English and español in my conversations
search mixing languages
recent 3

# Check your profile
profile

# Show stats
stats

# Get help
help
```

### 4. Python Script Testing

Create your own test script:
```python
from codeswitch_ai import EnhancedCodeSwitchDetector

# Initialize detector
detector = EnhancedCodeSwitchDetector()

# Test with your languages
text = "Your multilingual text here"
result = detector.analyze_with_user_guidance(
    text, 
    user_languages=["english", "spanish"]  # Your languages
)

print(f"Detected: {result.detected_languages}")
print(f"Switches: {result.switch_points}")
print(f"Confidence: {result.confidence:.2%}")

# Show phrases
for phrase in result.phrases:
    print(f"'{phrase.text}' → {phrase.language}")
```

## 📊 What to Test

### Language Combinations
- **English-Spanish**: "Hello, ¿cómo estás?"
- **Hindi-English**: "Main school ja raha hoon"  
- **Urdu-English**: "Aap kaise hain? How are you?"
- **French-English**: "Je suis tired aujourd'hui"
- **Arabic-English**: "Ana very happy today"

### Special Features
- **Function Words**: "The gato is sleeping"
- **Romanization**: "Inshallah everything will be fine"
- **User Guidance**: Set your languages first for better accuracy
- **Memory**: Store and search conversations

### Expected Results
- **Switch Points**: Should identify language change positions
- **Phrase Clustering**: Groups consecutive words in same language
- **User Language Match**: Higher accuracy when your languages are detected
- **Romanization Detection**: Identifies romanized Hindi/Urdu/Arabic

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install langdetect sentence-transformers faiss-cpu
   ```

2. **Low Accuracy**
   - Set your user languages: `set-languages english,spanish`
   - Use longer text passages for better context
   - Try function words: "the", "and", "el", "la", etc.

3. **No Switch Points**
   - Make sure languages are actually different
   - Check confidence thresholds (should be >40%)
   - Try with clearer language boundaries

### Performance Tips
- First run downloads sentence-transformer model (~90MB)
- Caching improves performance on repeated text
- User language guidance significantly improves accuracy

## 📈 Testing Different Scenarios

### 1. Accuracy Testing
```python
# Test with known language pairs
test_cases = [
    ("Hello, ¿cómo estás?", ["en", "es"]),
    ("Je suis tired", ["fr", "en"]), 
    ("Main ghar ja raha hoon", ["hi", "en"])
]

for text, expected_langs in test_cases:
    result = detector.analyze_with_user_guidance(text, expected_langs)
    detected = set(result.detected_languages)
    expected = set(expected_langs)
    accuracy = len(detected & expected) / len(expected | detected)
    print(f"Text: {text}")
    print(f"Expected: {expected_langs}, Got: {result.detected_languages}")
    print(f"Accuracy: {accuracy:.2%}")
```

### 2. Memory Testing
```python
from codeswitch_ai import ConversationMemory, EmbeddingGenerator

memory = ConversationMemory()
embedder = EmbeddingGenerator()

# Store some conversations
texts = [
    "Hello, ¿cómo estás?",
    "Je suis tired today", 
    "Main ghar ja raha hoon"
]

for text in texts:
    # Analyze and store...
```

### 3. Performance Testing
```python
import time

texts = ["Hello world"] * 100
start = time.time()

for text in texts:
    result = detector.analyze_with_user_guidance(text)

print(f"Processed {len(texts)} texts in {time.time() - start:.2f}s")
print(f"Average: {(time.time() - start) / len(texts) * 1000:.1f}ms per text")
```

## ✨ Expected Features Working

After successful testing, you should see:

✅ **Language Detection**: Identifies 10+ languages including romanized  
✅ **Switch Point Detection**: Finds language change boundaries  
✅ **Phrase Clustering**: Groups words by language  
✅ **User Guidance**: Better accuracy with your languages  
✅ **Romanization Support**: Detects romanized Hindi/Urdu/Arabic  
✅ **Function Word Recognition**: High accuracy on common words  
✅ **Memory Storage**: Persistent conversation storage  
✅ **Similarity Search**: Find similar conversations  
✅ **CLI Interface**: Interactive testing environment  

## 🎯 Success Criteria

The library is working correctly if:
- Examples run without errors
- CLI interface responds to commands  
- Switch points are detected in mixed-language text
- User language guidance improves results
- Memory features store and retrieve conversations
- Romanized text is detected (e.g., "main", "aap", "inshallah")

Try different language combinations and see how the enhanced detector performs compared to basic language detection!