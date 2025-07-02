# SwitchPrint Examples and Use Cases

This document provides comprehensive examples for SwitchPrint v2.1.1, combining both basic and advanced functionality.

**Current Status**: Production-ready with 100% test coverage  
**Installation**: `pip install switchprint==2.1.1`

## 🚀 Quick Start Examples

### Basic Language Detection

```python
from codeswitch_ai import EnsembleDetector

# Initialize with FastText only for speed
detector = EnsembleDetector(
    use_fasttext=True,
    use_transformer=False,
    ensemble_strategy="weighted_average"
)

# Simple detection
result = detector.detect_language("Hello, ¿cómo estás? I'm doing bien.")

print(f"Detected languages: {result.detected_languages}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Method: {result.method}")

```
## ⚡ High-Speed Detection (FastText Only)

```python
from codeswitch_ai import FastTextDetector

detector = FastTextDetector()
result = detector.detect_language("Main ghar ja raha hoon but I will come back")

print(f"Detected languages: {result.detected_languages}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Probabilities: {getattr(result, 'probabilities', 'N/A')}")

```

### 📊 Advanced Configuration

## 🧮 Threshold Mode Analysis

```python
from codeswitch_ai import EnsembleDetector, ThresholdConfig, DetectionMode

modes = [
    (DetectionMode.HIGH_PRECISION, "High Precision"),
    (DetectionMode.BALANCED, "Balanced"),
    (DetectionMode.HIGH_RECALL, "High Recall")
]

text = "Je suis tired today. Need some café."

for mode, description in modes:
    config = ThresholdConfig(mode=mode)
    detector = EnsembleDetector(threshold_config=config)
    result = detector.detect_language(text)
    print(f"{description:15} | {', '.join(result.detected_languages):15} | {result.confidence:.1%}")

```

## 🎯 Custom Threshold Profiles

```python
from codeswitch_ai import ThresholdProfile, ThresholdConfig, EnsembleDetector

profiles = [
    ThresholdProfile(
        name="Conservative",
        monolingual_min_confidence=0.9,
        multilingual_primary_confidence=0.8
    ),
    ThresholdProfile(
        name="Aggressive",
        monolingual_min_confidence=0.5,
        multilingual_primary_confidence=0.4
    )
]

for profile in profiles:
    config = ThresholdConfig(custom_profile=profile)
    detector = EnsembleDetector(threshold_config=config)
    result = detector.detect_language("Maybe this is english or maybe not")
    print(f"{profile.name:12} | {', '.join(result.detected_languages):15} | {result.confidence:.1%}")

```
### 🔍 Real-World Use Cases

## 🗣️ Multilingual Conversation Analysis

```python
from codeswitch_ai import EnsembleDetector, ConversationMemory

# Initialize components
detector = EnsembleDetector()
memory = ConversationMemory(db_path="conversations.db")

# Store and analyze conversations
conversations = [
    "Hello, ¿cómo estás? I'm doing bien today.",
    "Je suis tired but need to trabajo.",
    "Main office ja raha hoon, see you later."
]

for text in conversations:
    result = detector.detect_language(text)
    memory.create_and_store_conversation(
        text=text,
        user_id='demo_user',
        switch_stats={
            'detected_languages': result.detected_languages,
            'confidence': result.confidence
        }
    )

```

## 🛡️ Enterprise Security Pipeline

```python
from codeswitch_ai import (
    PrivacyProtector, 
    SecurityMonitor, 
    InputValidator,
    FastTextDetector
)

# Initialize security components
privacy_protector = PrivacyProtector()
security_monitor = SecurityMonitor()
input_validator = InputValidator()
detector = FastTextDetector()

text = "Hello, my email is john@example.com"

# Full processing pipeline
validation = input_validator.validate(text)
if validation.is_valid:
    privacy_result = privacy_protector.protect_text(validation.sanitized_text)
    detection_result = detector.detect_language(privacy_result['protected_text'])
    security_monitor.process_request(
        source_id="demo_request",
        request_data={'text_length': len(text)},
        user_id="demo_user"
    )

```

### ⚙️ Performance Optimization

## ⏱️ Benchmarking Different Detectors

```python
import time
from codeswitch_ai import FastTextDetector, EnsembleDetector

detectors = [
    ("FastText", FastTextDetector()),
    ("Ensemble", EnsembleDetector(use_fasttext=True, use_transformer=False))
]

test_text = "Hello, ¿cómo estás? Je suis tired today."
iterations = 100

for name, detector in detectors:
    start = time.time()
    for _ in range(iterations):
        detector.detect_language(test_text)
    avg_time = (time.time() - start) / iterations
    print(f"{name:12} | {avg_time*1000:6.2f}ms per detection")

```

## 🧠 Ensemble Strategy Comparison

```python
from codeswitch_ai import EnsembleDetector

strategies = ["weighted_average", "voting", "confidence_based"]
test_text = "Hello, ¿cómo estás? Je suis bien."

for strategy in strategies:
    detector = EnsembleDetector(ensemble_strategy=strategy)
    result = detector.detect_language(test_text)
    print(f"{strategy:18} | {', '.join(result.detected_languages):20} | {result.confidence:.1%}")

```

### 🔧 Configuration Examples
## 🏗️ Production-Ready Setup

```python
from codeswitch_ai import (
    EnsembleDetector,
    PrivacyProtector,
    SecurityMonitor,
    ThresholdConfig,
    DetectionMode
)

# Production configuration
detector = EnsembleDetector(
    use_fasttext=True,
    use_transformer=False,
    threshold_config=ThresholdConfig(mode=DetectionMode.HIGH_PRECISION),
    cache_size=5000
)

privacy_protector = PrivacyProtector()
security_monitor = SecurityMonitor(log_file="security.log")

```

### 📚 Integration Examples
## 🌐 Flask API Endpoint

```python
from flask import Flask, request, jsonify
from codeswitch_ai import EnsembleDetector

app = Flask(__name__)
detector = EnsembleDetector()

@app.route('/detect', methods=['POST'])
def detect():
    text = request.json.get('text', '')
    result = detector.detect_language(text)
    return jsonify({
        'languages': result.detected_languages,
        'confidence': result.confidence,
        'method': result.method
    })

if __name__ == '__main__':
    app.run()

```

## 💻 Command Line Interface

```python
import argparse
from codeswitch_ai import FastTextDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text', help='Text to analyze')
    args = parser.parse_args()
    
    detector = FastTextDetector()
    result = detector.detect_language(args.text)
    
    print(f"Detected languages: {', '.join(result.detected_languages)}")
    print(f"Confidence: {result.confidence:.1%}")

if __name__ == '__main__':
    main()

```

### Advanced Detection with User Context

```python
from codeswitch_ai import EnsembleDetector

detector = EnsembleDetector(
    use_fasttext=True,
    use_transformer=True,
    ensemble_strategy="weighted_average"
)

# Detection with user language context
result = detector.detect_language(
    text="Je suis très tired aujourd'hui",
    user_languages=["french", "english"]
)

print(f"Detected: {result.detected_languages}")
print(f"Confidence boost from user context: {result.confidence:.2%}")
```

## 📊 Real-World Use Cases

### 1. Social Media Analysis

**Scenario**: Analyze code-switching patterns in multilingual social media posts.

```python
from codeswitch_ai import EnsembleDetector, ConversationMemory
import pandas as pd

# Initialize components
detector = EnsembleDetector()
memory = ConversationMemory()

# Sample social media posts
posts = [
    "Just had the best café con leche ☕ #MorningVibes",
    "Going to the biblioteca to study. Wish me luck! 📚",
    "Мой друг said this movie is amazing. Can't wait to watch!",
    "Planning a road trip avec mes amis. So excited! 🚗"
]

# Analyze posts
results = []
for i, post in enumerate(posts):
    result = detector.detect_language(post)
    
    # Store in memory for pattern analysis
    memory.store_conversation(
        user_id=f"user_{i}",
        text=post,
        metadata={
            'platform': 'twitter',
            'detected_languages': result.detected_languages,
            'confidence': result.confidence
        }
    )
    
    results.append({
        'post': post,
        'languages': result.detected_languages,
        'confidence': result.confidence,
        'switch_count': len(result.switch_points)
    })

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
print(df)
```

### 2. Customer Support Multilingual Chat

**Scenario**: Detect language switches in customer support conversations to route to appropriate agents.

```python
from codeswitch_ai import (
    EnsembleDetector, PrivacyProtector, SecurityMonitor,
    InputValidator, PrivacyLevel
)

class MultilingualChatSupport:
    def __init__(self):
        self.detector = EnsembleDetector()
        self.privacy_protector = PrivacyProtector(
            config=PrivacyConfig(privacy_level=PrivacyLevel.HIGH)
        )
        self.security_monitor = SecurityMonitor()
        self.validator = InputValidator()
    
    def process_customer_message(self, message: str, customer_id: str):
        """Process customer message with security and privacy protection."""
        
        # 1. Input validation
        validation = self.validator.validate(message)
        if not validation.is_valid:
            return {
                'error': 'Invalid input',
                'threats': validation.threats_detected
            }
        
        # 2. Privacy protection (remove PII)
        privacy_result = self.privacy_protector.protect_text(
            validation.sanitized_text,
            source_id=customer_id
        )
        
        # 3. Language detection
        detection_result = self.detector.detect_language(
            privacy_result['protected_text']
        )
        
        # 4. Security monitoring
        security_events = self.security_monitor.process_request(
            source_id='customer_chat',
            request_data={
                'text_size': len(message),
                'detected_languages': detection_result.detected_languages
            },
            user_id=customer_id
        )
        
        # 5. Route to appropriate agent
        primary_language = detection_result.detected_languages[0]
        agent_routing = self.route_to_agent(primary_language)
        
        return {
            'detected_languages': detection_result.detected_languages,
            'confidence': detection_result.confidence,
            'switch_points': len(detection_result.switch_points),
            'agent_routing': agent_routing,
            'privacy_applied': privacy_result['protection_applied'],
            'security_events': len(security_events),
            'processed_text': privacy_result['protected_text']
        }
    
    def route_to_agent(self, primary_language: str) -> dict:
        """Route customer to appropriate language support agent."""
        language_agents = {
            'english': 'EN_SUPPORT_QUEUE',
            'spanish': 'ES_SUPPORT_QUEUE',
            'french': 'FR_SUPPORT_QUEUE',
            'chinese': 'CN_SUPPORT_QUEUE'
        }
        
        return {
            'queue': language_agents.get(primary_language, 'MULTILINGUAL_QUEUE'),
            'priority': 'high' if primary_language not in language_agents else 'normal'
        }

# Usage example
support_system = MultilingualChatSupport()

customer_messages = [
    "Hello, I need help with my order. Mi pedido no llegó.",
    "Bonjour, j'ai un problème avec my account settings.",
    "My password is not working. 密码不起作用。"
]

for i, message in enumerate(customer_messages):
    result = support_system.process_customer_message(message, f"customer_{i}")
    print(f"Customer {i}: {result['agent_routing']['queue']}")
    print(f"Languages: {result['detected_languages']}")
    print(f"Privacy protection: {result['privacy_applied']}")
    print("---")
```

### 3. Educational Language Learning Assessment

**Scenario**: Assess language learning progress by analyzing code-switching patterns in student writing.

```python
from codeswitch_ai import (
    EnsembleDetector, TemporalCodeSwitchAnalyzer, 
    ConversationMemory
)
import datetime

class LanguageLearningAssessment:
    def __init__(self):
        self.detector = EnsembleDetector()
        self.memory = ConversationMemory()
        self.temporal_analyzer = TemporalCodeSwitchAnalyzer(self.memory)
    
    def assess_student_writing(self, student_id: str, text: str, 
                             target_language: str, native_language: str):
        """Assess student's language learning progress."""
        
        # Detect code-switching patterns
        result = self.detector.detect_language(
            text,
            user_languages=[target_language, native_language]
        )
        
        # Store for temporal analysis
        self.memory.store_conversation(
            user_id=student_id,
            text=text,
            metadata={
                'target_language': target_language,
                'native_language': native_language,
                'assessment_date': datetime.datetime.now().isoformat(),
                'detected_languages': result.detected_languages,
                'confidence': result.confidence
            }
        )
        
        # Calculate language proficiency metrics
        target_language_ratio = self._calculate_language_ratio(
            result, target_language
        )
        
        switch_density = len(result.switch_points) / len(text.split())
        
        # Assess fluency based on switching patterns
        fluency_score = self._assess_fluency(
            target_language_ratio, switch_density, result.confidence
        )
        
        return {
            'student_id': student_id,
            'target_language_ratio': target_language_ratio,
            'switch_density': switch_density,
            'fluency_score': fluency_score,
            'detected_languages': result.detected_languages,
            'switch_points': result.switch_points,
            'recommendations': self._generate_recommendations(
                target_language_ratio, switch_density
            )
        }
    
    def get_progress_report(self, student_id: str, days: int = 30):
        """Generate progress report for student."""
        temporal_stats = self.temporal_analyzer.analyze_user_patterns(
            student_id, time_range_days=days
        )
        
        return {
            'student_id': student_id,
            'time_period': f"{days} days",
            'total_submissions': temporal_stats.total_conversations,
            'language_distribution': temporal_stats.language_distribution,
            'switching_frequency_trend': temporal_stats.switching_frequency_over_time,
            'improvement_score': self._calculate_improvement_score(temporal_stats)
        }
    
    def _calculate_language_ratio(self, result, target_language: str) -> float:
        """Calculate ratio of target language usage."""
        target_count = sum(
            1 for phrase in result.phrases 
            if phrase['language'] == target_language
        )
        total_phrases = len(result.phrases)
        return target_count / total_phrases if total_phrases > 0 else 0.0
    
    def _assess_fluency(self, target_ratio: float, switch_density: float, 
                       confidence: float) -> float:
        """Assess fluency based on multiple factors."""
        # Higher target language ratio = better fluency
        ratio_score = target_ratio * 0.4
        
        # Lower switch density = better fluency
        switch_score = max(0, (1 - switch_density) * 0.3)
        
        # Higher confidence = better fluency
        confidence_score = confidence * 0.3
        
        return ratio_score + switch_score + confidence_score
    
    def _generate_recommendations(self, target_ratio: float, 
                                switch_density: float) -> list:
        """Generate learning recommendations."""
        recommendations = []
        
        if target_ratio < 0.5:
            recommendations.append(
                "Try to use more of the target language in your writing"
            )
        
        if switch_density > 0.2:
            recommendations.append(
                "Focus on completing thoughts in one language before switching"
            )
        
        if not recommendations:
            recommendations.append("Great job! Keep practicing to maintain fluency")
        
        return recommendations
    
    def _calculate_improvement_score(self, temporal_stats) -> float:
        """Calculate improvement score over time."""
        # Implementation would analyze temporal trends
        # This is a simplified version
        return 0.75  # Placeholder

# Usage example
assessment_system = LanguageLearningAssessment()

# Sample student writings
student_writings = [
    {
        'student_id': 'student_001',
        'text': "I went to the mercado today and bought algunas frutas. The apples were very fresh.",
        'target_language': 'english',
        'native_language': 'spanish'
    },
    {
        'student_id': 'student_001', 
        'text': "Yesterday I studied for my examen de matemáticas. It was quite difficult but I think I did well.",
        'target_language': 'english',
        'native_language': 'spanish'
    }
]

for writing in student_writings:
    assessment = assessment_system.assess_student_writing(**writing)
    print(f"Student: {assessment['student_id']}")
    print(f"Fluency Score: {assessment['fluency_score']:.2f}")
    print(f"Target Language Ratio: {assessment['target_language_ratio']:.2%}")
    print(f"Recommendations: {assessment['recommendations']}")
    print("---")

# Generate progress report
progress = assessment_system.get_progress_report('student_001')
print(f"Progress Report: {progress}")
```

### 4. Content Moderation for Multilingual Platforms

**Scenario**: Moderate content across multiple languages while detecting potentially harmful code-switching patterns.

```python
from codeswitch_ai import (
    EnsembleDetector, SecurityMonitor, InputValidator, 
    PrivacyProtector, SecurityConfig
)

class MultilingualContentModerator:
    def __init__(self):
        self.detector = EnsembleDetector()
        self.security_monitor = SecurityMonitor()
        self.input_validator = InputValidator(
            config=SecurityConfig(security_level='strict')
        )
        self.privacy_protector = PrivacyProtector()
        
        # Content moderation rules by language
        self.moderation_rules = {
            'english': ['spam', 'hate', 'violence'],
            'spanish': ['spam', 'odio', 'violencia'],
            'french': ['spam', 'haine', 'violence']
        }
    
    def moderate_content(self, content: str, user_id: str, 
                        platform: str = 'forum') -> dict:
        """Moderate multilingual content."""
        
        # 1. Input validation and security check
        validation = self.input_validator.validate(content)
        if not validation.is_valid:
            return {
                'action': 'reject',
                'reason': 'Security violation',
                'details': validation.threats_detected
            }
        
        # 2. Language detection
        detection_result = self.detector.detect_language(
            validation.sanitized_text
        )
        
        # 3. Privacy protection (detect and flag PII)
        privacy_result = self.privacy_protector.protect_text(
            validation.sanitized_text,
            source_id=user_id
        )
        
        # 4. Content analysis by language
        moderation_flags = self._check_content_rules(
            content, detection_result.detected_languages
        )
        
        # 5. Security monitoring
        security_events = self.security_monitor.process_request(
            source_id=f'{platform}_moderation',
            request_data={
                'text_size': len(content),
                'detected_languages': detection_result.detected_languages,
                'moderation_flags': len(moderation_flags),
                'pii_detected': len(privacy_result['pii_detected'])
            },
            user_id=user_id
        )
        
        # 6. Make moderation decision
        decision = self._make_moderation_decision(
            moderation_flags, 
            privacy_result,
            detection_result,
            security_events
        )
        
        return decision
    
    def _check_content_rules(self, content: str, languages: list) -> list:
        """Check content against language-specific moderation rules."""
        flags = []
        content_lower = content.lower()
        
        for language in languages:
            if language in self.moderation_rules:
                for rule in self.moderation_rules[language]:
                    if rule in content_lower:
                        flags.append({
                            'language': language,
                            'rule': rule,
                            'severity': 'medium'
                        })
        
        return flags
    
    def _make_moderation_decision(self, moderation_flags: list,
                                privacy_result: dict,
                                detection_result,
                                security_events: list) -> dict:
        """Make final moderation decision."""
        
        # Calculate risk score
        risk_score = 0.0
        
        # Content flags
        risk_score += len(moderation_flags) * 0.3
        
        # Privacy violations
        if privacy_result['pii_detected']:
            risk_score += len(privacy_result['pii_detected']) * 0.2
        
        # Security events
        risk_score += len(security_events) * 0.4
        
        # Language switching patterns (excessive switching might indicate spam)
        if len(detection_result.switch_points) > 10:
            risk_score += 0.1
        
        # Decision logic
        if risk_score >= 1.0:
            action = 'reject'
            reason = 'High risk content detected'
        elif risk_score >= 0.5:
            action = 'review'
            reason = 'Requires human review'
        else:
            action = 'approve'
            reason = 'Content approved'
        
        return {
            'action': action,
            'reason': reason,
            'risk_score': risk_score,
            'detected_languages': detection_result.detected_languages,
            'moderation_flags': moderation_flags,
            'pii_detected': len(privacy_result['pii_detected']),
            'security_events': len(security_events)
        }

# Usage example
moderator = MultilingualContentModerator()

content_samples = [
    "Great post! Me gusta mucho this discussion.",
    "This is spam spam spam in multiple languages.",
    "My email is john@example.com and phone is 555-1234.",
    "Normal content discussing el clima and weather patterns."
]

for i, content in enumerate(content_samples):
    decision = moderator.moderate_content(content, f"user_{i}")
    print(f"Content: {content[:50]}...")
    print(f"Decision: {decision['action']} - {decision['reason']}")
    print(f"Risk Score: {decision['risk_score']:.2f}")
    print(f"Languages: {decision['detected_languages']}")
    print("---")
```

### 5. Research Data Analysis

**Scenario**: Analyze large-scale multilingual datasets for sociolinguistic research.

```python
from codeswitch_ai import (
    EnsembleDetector, TemporalCodeSwitchAnalyzer,
    LinCEBenchmark, MTEBEvaluator, ConversationMemory
)
import pandas as pd
import matplotlib.pyplot as plt

class SociolinguisticResearchTool:
    def __init__(self):
        self.detector = EnsembleDetector()
        self.memory = ConversationMemory()
        self.temporal_analyzer = TemporalCodeSwitchAnalyzer(self.memory)
        self.lince_benchmark = LinCEBenchmark()
        
    def analyze_corpus(self, corpus_data: pd.DataFrame) -> dict:
        """Analyze a multilingual corpus for research insights."""
        
        results = {
            'corpus_stats': {},
            'language_patterns': {},
            'switching_patterns': {},
            'demographic_analysis': {}
        }
        
        # Process each text in corpus
        processed_data = []
        for idx, row in corpus_data.iterrows():
            text = row['text']
            user_id = row.get('user_id', f'user_{idx}')
            demographics = row.get('demographics', {})
            
            # Detect code-switching
            detection_result = self.detector.detect_language(text)
            
            # Store in memory for temporal analysis
            self.memory.store_conversation(
                user_id=user_id,
                text=text,
                metadata={
                    'demographics': demographics,
                    'detected_languages': detection_result.detected_languages,
                    'confidence': detection_result.confidence,
                    'switch_count': len(detection_result.switch_points)
                }
            )
            
            processed_data.append({
                'user_id': user_id,
                'text': text,
                'languages': detection_result.detected_languages,
                'switch_points': len(detection_result.switch_points),
                'confidence': detection_result.confidence,
                'demographics': demographics
            })
        
        # Analyze patterns
        results['corpus_stats'] = self._analyze_corpus_statistics(processed_data)
        results['language_patterns'] = self._analyze_language_patterns(processed_data)
        results['switching_patterns'] = self._analyze_switching_patterns(processed_data)
        results['demographic_analysis'] = self._analyze_demographics(processed_data)
        
        return results
    
    def _analyze_corpus_statistics(self, data: list) -> dict:
        """Analyze basic corpus statistics."""
        total_texts = len(data)
        total_switches = sum(item['switch_points'] for item in data)
        avg_confidence = sum(item['confidence'] for item in data) / total_texts
        
        # Language distribution
        all_languages = []
        for item in data:
            all_languages.extend(item['languages'])
        
        language_counts = pd.Series(all_languages).value_counts()
        
        return {
            'total_texts': total_texts,
            'total_switches': total_switches,
            'avg_switches_per_text': total_switches / total_texts,
            'avg_confidence': avg_confidence,
            'language_distribution': language_counts.to_dict(),
            'multilingual_texts': sum(1 for item in data if len(item['languages']) > 1)
        }
    
    def _analyze_language_patterns(self, data: list) -> dict:
        """Analyze language combination patterns."""
        language_pairs = {}
        language_combinations = {}
        
        for item in data:
            languages = sorted(item['languages'])
            if len(languages) > 1:
                # Language pairs
                for i in range(len(languages)):
                    for j in range(i+1, len(languages)):
                        pair = (languages[i], languages[j])
                        language_pairs[pair] = language_pairs.get(pair, 0) + 1
                
                # Language combinations
                combo = tuple(languages)
                language_combinations[combo] = language_combinations.get(combo, 0) + 1
        
        return {
            'most_common_pairs': sorted(language_pairs.items(), key=lambda x: x[1], reverse=True)[:10],
            'most_common_combinations': sorted(language_combinations.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _analyze_switching_patterns(self, data: list) -> dict:
        """Analyze code-switching patterns."""
        switch_counts = [item['switch_points'] for item in data]
        
        return {
            'switch_distribution': pd.Series(switch_counts).describe().to_dict(),
            'high_switchers': sum(1 for count in switch_counts if count > 5),
            'no_switch_texts': sum(1 for count in switch_counts if count == 0)
        }
    
    def _analyze_demographics(self, data: list) -> dict:
        """Analyze patterns by demographics."""
        demographic_patterns = {}
        
        # Group by age if available
        age_groups = {}
        for item in data:
            age = item['demographics'].get('age')
            if age:
                age_group = f"{(age//10)*10}s"
                if age_group not in age_groups:
                    age_groups[age_group] = []
                age_groups[age_group].append(item['switch_points'])
        
        for age_group, switches in age_groups.items():
            demographic_patterns[f'age_{age_group}'] = {
                'avg_switches': sum(switches) / len(switches),
                'total_speakers': len(switches)
            }
        
        return demographic_patterns
    
    def generate_research_report(self, analysis_results: dict) -> str:
        """Generate formatted research report."""
        
        report = """
# Sociolinguistic Analysis Report

## Corpus Statistics
- Total texts analyzed: {total_texts}
- Multilingual texts: {multilingual_texts} ({multilingual_pct:.1f}%)
- Average switches per text: {avg_switches:.2f}
- Average detection confidence: {avg_confidence:.2%}

## Language Distribution
{language_dist}

## Most Common Language Pairs
{language_pairs}

## Code-Switching Patterns
- Mean switches: {switch_mean:.2f}
- Standard deviation: {switch_std:.2f}
- High-switching texts (>5 switches): {high_switchers}

## Demographic Insights
{demographic_insights}
        """.format(
            total_texts=analysis_results['corpus_stats']['total_texts'],
            multilingual_texts=analysis_results['corpus_stats']['multilingual_texts'],
            multilingual_pct=(analysis_results['corpus_stats']['multilingual_texts'] / 
                             analysis_results['corpus_stats']['total_texts'] * 100),
            avg_switches=analysis_results['corpus_stats']['avg_switches_per_text'],
            avg_confidence=analysis_results['corpus_stats']['avg_confidence'],
            language_dist='\n'.join([f"- {lang}: {count}" for lang, count in 
                                   list(analysis_results['corpus_stats']['language_distribution'].items())[:5]]),
            language_pairs='\n'.join([f"- {pair[0]}-{pair[1]}: {count}" for pair, count in 
                                    analysis_results['language_patterns']['most_common_pairs'][:5]]),
            switch_mean=analysis_results['switching_patterns']['switch_distribution']['mean'],
            switch_std=analysis_results['switching_patterns']['switch_distribution']['std'],
            high_switchers=analysis_results['switching_patterns']['high_switchers'],
            demographic_insights=str(analysis_results['demographic_analysis'])
        )
        
        return report

# Usage example
research_tool = SociolinguisticResearchTool()

# Sample research corpus
corpus_data = pd.DataFrame([
    {
        'text': "I love this ciudad, it's so beautiful and the people are muy amables.",
        'user_id': 'participant_001',
        'demographics': {'age': 25, 'native_language': 'spanish', 'education': 'university'}
    },
    {
        'text': "Going to work tomorrow, mais je suis très tired already.",
        'user_id': 'participant_002', 
        'demographics': {'age': 30, 'native_language': 'french', 'education': 'graduate'}
    },
    {
        'text': "The weather today is perfect for walking in the park.",
        'user_id': 'participant_003',
        'demographics': {'age': 35, 'native_language': 'english', 'education': 'university'}
    }
])

# Analyze corpus
analysis_results = research_tool.analyze_corpus(corpus_data)

# Generate report
report = research_tool.generate_research_report(analysis_results)
print(report)
```

## 🔧 Configuration Examples

### Production Configuration

```python
# production_config.py
from codeswitch_ai import (
    EnsembleDetector, SecurityConfig, PrivacyConfig, 
    PrivacyLevel, SecurityMonitor
)

# Production-ready configuration
PRODUCTION_CONFIG = {
    'detector': {
        'use_fasttext': True,
        'use_transformer': True,
        'ensemble_strategy': 'weighted_average',
        'cache_size': 5000
    },
    'security': SecurityConfig(
        security_level='strict',
        max_text_length=50000,
        enable_html_sanitization=True,
        enable_injection_detection=True,
        enable_pii_detection=True
    ),
    'privacy': PrivacyConfig(
        privacy_level=PrivacyLevel.HIGH,
        anonymization_method='replacement',
        preserve_language_structure=True
    ),
    'monitoring': {
        'log_file': '/var/log/switchprint/security.log',
        'enable_metrics': True,
        'alert_threshold': 'medium'
    }
}

def create_production_detector():
    """Create production-ready detector with security."""
    detector = EnsembleDetector(**PRODUCTION_CONFIG['detector'])
    security_monitor = SecurityMonitor(PRODUCTION_CONFIG['monitoring']['log_file'])
    
    return detector, security_monitor
```

## 📚 Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
from codeswitch_ai import EnsembleDetector, PrivacyProtector

app = Flask(__name__)
detector = EnsembleDetector()
privacy_protector = PrivacyProtector()

@app.route('/api/detect', methods=['POST'])
def detect_language():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Apply privacy protection
    privacy_result = privacy_protector.protect_text(text)
    
    # Detect languages
    result = detector.detect_language(privacy_result['protected_text'])
    
    return jsonify({
        'languages': result.detected_languages,
        'confidence': result.confidence,
        'switch_points': result.switch_points,
        'privacy_applied': privacy_result['protection_applied']
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### Streamlit Dashboard

```python
import streamlit as st
from codeswitch_ai import EnsembleDetector, ConversationMemory
import plotly.express as px
import pandas as pd

# Initialize components
if 'detector' not in st.session_state:
    st.session_state.detector = EnsembleDetector()
    st.session_state.memory = ConversationMemory()

st.title("SwitchPrint - Code-Switching Analysis Dashboard")

# Text input
text = st.text_area("Enter multilingual text:", height=100)

if st.button("Analyze"):
    if text:
        # Detect languages
        result = st.session_state.detector.detect_language(text)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detection Results")
            st.write(f"**Languages:** {', '.join(result.detected_languages)}")
            st.write(f"**Confidence:** {result.confidence:.2%}")
            st.write(f"**Switch Points:** {len(result.switch_points)}")
        
        with col2:
            st.subheader("Language Distribution")
            if result.phrases:
                phrase_data = pd.DataFrame(result.phrases)
                fig = px.pie(phrase_data, names='language', title="Language Distribution")
                st.plotly_chart(fig)
        
        # Display phrases
        st.subheader("Phrase Analysis")
        for phrase in result.phrases:
            st.write(f"**{phrase['language']}**: '{phrase['text']}' (confidence: {phrase['confidence']:.2%})")
```

These examples demonstrate the versatility and power of SwitchPrint across various domains and use cases, from simple language detection to complex multilingual analysis systems.