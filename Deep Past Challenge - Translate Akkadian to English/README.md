# Deep Past Challenge: Akkadian to English Neural Machine Translation

## 🎯 Challenge Overview

This project tackles the challenging task of translating **transliterated Akkadian** (an ancient Mesopotamian language) into **English**. Akkadian is a low-resource, morphologically complex language where a single word can encode what takes multiple words in English.

## 📊 Dataset

- **Training Set**: 1,200+ parallel Akkadian-English sentence pairs
- **Test Set**: 3 test examples for prediction
- **Lexicon**: 3,500+ Akkadian word entries from electronic Babylonian Library (eBL)
- **Publications**: ~900 OCR'd PDFs with multilingual translations (German, French, Turkish, English)
- **Published Texts**: 11,000+ tablet transliterations with metadata
- **Domain**: Ancient administrative documents and personal correspondence (Old Assyrian period)

### Data Extraction Challenge

The publications.csv contains OCR output from ~900 scholarly PDFs. Before training, you need to:
1. **Extract translations** from mixed-language OCR text
2. **Match** transliterations with their translations using document IDs
3. **Normalize** all translations to English (from German, French, Turkish, etc.)
4. **Align** at sentence level for optimal training

Our notebook includes a complete data reconstruction pipeline for this!

### Key Statistics:
- **Average Akkadian sentence length**: 57.5 words
- **Average English translation length**: 90.5 words
- **Translation ratio**: ~1.43 (English uses ~43% more words than Akkadian)
- This demonstrates Akkadian's morphological richness!

## 🏗️ Architecture & Approach

### Phase 1: Data Reconstruction (Sections 2A-2D)

**Critical preprocessing** before any ML training:

1. **Publication Translation Extraction**
   - Parse OCR text from ~900 scholarly PDFs
   - Identify document IDs and match with transliterations
   - Extract quoted translations using regex patterns
   - Handle multiple scholarly notation systems

2. **Multilingual Translation Normalization**
   - Detect language (German, French, Turkish, English)
   - Use Helsinki-NLP translation models (Marian)
   - Convert all translations to English
   - Preserve semantic meaning across languages

3. **Sentence-Level Alignment**
   - Split Akkadian using linguistic markers (-ma, um-ma, etc.)
   - Split English using punctuation
   - Align using length ratio heuristics
   - Create parallel sentence pairs for training

4. **Data Augmentation from Reconstructed Corpus**
   - Merge extracted translations with existing training data
   - Deduplicate based on Akkadian content
   - Create enhanced training corpus

### Phase 2: Neural Translation Models

### Models Implemented:

1. **T5 (Text-to-Text Transfer Transformer)**
   - Base model: `t5-small` (60M parameters)
   - Fine-tuned on Akkadian-English pairs
   - Uses prefix: "translate Akkadian to English: [text]"
   
2. **MarianMT (Multilingual Translation)**
   - Specialized translation architecture
   - Pre-trained on multiple language pairs
   - Fast inference with optimized beam search

3. **Custom Transformer with Morphological Attention**
   - Custom attention mechanism for morphological features
   - Positional encoding for sequence understanding
   - Designed specifically for Akkadian's complex structure

### Key Techniques for Low-Resource NMT:

#### 1. **Transfer Learning**
- Start with pre-trained models (T5, mBART, mT5)
- Fine-tune on domain-specific Akkadian data
- Leverage knowledge from high-resource language pairs

#### 2. **Subword Tokenization**
- Byte-Pair Encoding (BPE) for morphological segmentation
- Handles Akkadian's agglutinative morphology
- Reduces vocabulary size while maintaining semantic information

#### 3. **Data Augmentation**
- Word dropout (randomly remove words)
- Word shuffling (local reordering)
- Synthetic example generation
- Back-translation (when possible)

#### 4. **Ensemble Methods**
- Combine predictions from multiple models
- Voting, reranking, and weighted averaging
- BLEU-score based selection

#### 5. **Curriculum Learning**
- Train on simpler examples first
- Gradually introduce more complex sentences
- Sort by length, translation ratio, or complexity

## 🔬 Technical Details

### Preprocessing Pipeline:
```python
1. Unicode normalization (handle special Akkadian characters)
2. Text cleaning (remove excessive whitespace, quotes)
3. Lowercase normalization
4. Special character mapping (á¹­ → t, á¸« → h, etc.)
```

### Training Configuration:
- **Optimizer**: AdamW
- **Learning Rate**: 3e-5 with linear warmup
- **Batch Size**: 8 (adjustable based on GPU)
- **Epochs**: 5
- **Max Sequence Length**: 512 tokens
- **Gradient Clipping**: 1.0
- **Dropout**: 0.1

### Inference Configuration:
- **Beam Search**: 4-5 beams
- **Length Penalty**: 1.0
- **No Repeat N-gram**: 3
- **Early Stopping**: Enabled

## 📈 Evaluation Metrics

### BLEU Score (Primary)
- Measures n-gram overlap between prediction and reference
- Industry standard for machine translation
- Ranges from 0 to 1 (higher is better)

### Custom Metrics:
- Length accuracy (how close to reference length)
- Vocabulary coverage (% of Akkadian words translated)
- Morphological consistency

## � Data Extraction & Reconstruction Pipeline

**⚠️ CRITICAL STEP**: Before training models, you must extract and align translations from the OCR dataset!

The `publications.csv` file (554MB) contains OCR text from ~900 scholarly PDFs with Akkadian transliterations and translations in multiple languages (English, German, French, Turkish). We've implemented a complete 4-step pipeline to extract this data:

### Pipeline Overview
```
publications.csv (554MB OCR) → Extraction → Normalization → Alignment → Training Data (2-3x expansion)
     ↓                              ↓              ↓              ↓
~900 PDFs                    Parse text      Translate      Match sentences
                             Match docs      to English     by length ratio
```

### Step-by-Step Process:

#### **Step 1: Publication Translation Extraction**
Extracts translations from OCR text using document matching and pattern recognition.

```python
from notebook import PublicationTranslationExtractor

extractor = PublicationTranslationExtractor()
translations_df = extractor.extract_translations(
    publications_df, 
    published_texts_df,
    sample_size=10  # Start small, then scale to full dataset
)
```

**Output**: DataFrame with columns: `[id, transliteration, extracted_translation, source_language, confidence]`

#### **Step 2: Multilingual Translation Normalization**
Converts German, French, and Turkish translations to English using MarianMT models.

```python
from notebook import MultilingualTranslationNormalizer

normalizer = MultilingualTranslationNormalizer()
normalized_df = normalizer.normalize_translations(translations_df)
```

**Output**: All translations converted to English with language detection metadata.

#### **Step 3: Sentence-Level Alignment**
Aligns Akkadian transliterations with English translations at the sentence level.

```python
from notebook import SentenceAligner

aligner = SentenceAligner()
aligned_df = aligner.align_sentences(normalized_df)
```

**Output**: Sentence-aligned pairs ready for NMT training.

#### **Step 4: Complete Reconstruction**
Run the entire pipeline with one command:

```python
from notebook import complete_data_reconstruction_pipeline

reconstruction_results = complete_data_reconstruction_pipeline(
    publications_df,
    published_texts_df,
    sample_size=50,        # Start with 50 publications for testing
    output_dir='./data/reconstructed'
)

# Results dictionary contains:
# - extracted_df: Raw extracted translations
# - normalized_df: Translations converted to English
# - aligned_df: Sentence-aligned pairs
# - final_training_df: Combined with train.csv
# - statistics: Processing metrics
```

### 📊 Expected Results

| Dataset | Before Extraction | After Extraction | Improvement |
|---------|------------------|------------------|-------------|
| Training Pairs | 1,200 | 2,400-3,600 | 2-3x |
| Unique Akkadian Words | ~5,000 | ~10,000-15,000 | 2-3x |
| Coverage | Limited | Comprehensive | Full corpus |

### ⏱️ Processing Time

| Step | Sample (n=50) | Full Dataset (n=900) | Notes |
|------|--------------|----------------------|-------|
| Extraction | 5-10 min | 4-6 hours | OCR parsing intensive |
| Normalization | 2-5 min | 2-4 hours | Requires Helsinki models |
| Alignment | 1-2 min | 1-2 hours | Sentence tokenization |
| **Total** | **10-20 min** | **8-12 hours** | One-time preprocessing |

### 💾 Memory Requirements

- **Sample Mode** (n=50): ~2-4 GB RAM
- **Full Dataset** (n=900): ~8-16 GB RAM
- **Recommendations**: 
  - Use chunked processing for systems with <16GB RAM
  - Cache intermediate results to avoid re-processing

### 📖 Detailed Documentation

For complete implementation details, troubleshooting, and advanced usage, see:
**[DATA_EXTRACTION_GUIDE.md](DATA_EXTRACTION_GUIDE.md)** - 3,500+ word comprehensive guide

### ⚙️ Configuration Options

```python
# Fine-tune extraction parameters
extractor = PublicationTranslationExtractor(
    min_confidence=0.6,           # Document matching threshold
    max_translation_length=500,   # Filter very long translations
    enable_fuzzy_matching=True    # Use fuzzy matching for titles
)

# Control normalization behavior
normalizer = MultilingualTranslationNormalizer(
    batch_size=32,               # Translation batch size
    detect_threshold=0.8,        # Language detection confidence
    fallback_to_english=True     # Keep English if detection fails
)

# Adjust alignment parameters
aligner = SentenceAligner(
    length_ratio_threshold=3.0,  # Max Akkadian/English length ratio
    min_sentence_length=5,       # Skip very short sentences
    use_linguistic_rules=True    # Apply Akkadian sentence rules
)
```

## 🚀 Quick Start

### **Phase 1: Data Reconstruction** (Run First!)

#### 1.1. Setup Environment
```python
# All dependencies are pre-installed in the notebook
import torch
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

#### 1.2. Load Raw Data
```python
# Load the three data files
publications_df = pd.read_csv('publications.csv')      # 554MB - OCR from PDFs
published_texts_df = pd.read_csv('published_texts.csv')  # 11,000+ texts
train_df = pd.read_csv('train.csv')                   # 1,200 pre-aligned pairs
test_df = pd.read_csv('test.csv')                     # Test set for submission
```

#### 1.3. Run Data Extraction Pipeline
```python
# CRITICAL: Extract translations from OCR before training!
reconstruction_results = complete_data_reconstruction_pipeline(
    publications_df,
    published_texts_df,
    sample_size=100,  # Start with 100, then scale up
    output_dir='./data/reconstructed'
)

# Use reconstructed data for training
train_df = reconstruction_results['final_training_df']
print(f"Training data expanded: {len(train_df)} pairs")
```

### **Phase 2: Model Training**

#### 2.1. Load Data
```python
# Use reconstructed data from Phase 1
train_df = pd.read_csv('./data/reconstructed/final_training_data.csv')
test_df = pd.read_csv('test.csv')
```

#### 2.2. Train Model
```python
t5_model = T5TranslationModel(model_name='t5-small', device=device)
train_dataset, val_dataset = t5_model.prepare_data(train_df, test_size=0.15)

history = t5_model.train(
    train_dataset, 
    val_dataset, 
    epochs=5, 
    batch_size=8, 
    lr=3e-5
)
```

#### 2.3. Generate Predictions
```python
submission = generate_submission(t5_model, test_df, 'submission.csv')
```

### **Complete Workflow Example**

Here's the entire pipeline from raw data to submission:

```python
# ============================================
# STEP 1: Data Reconstruction (8-12 hours)
# ============================================
from notebook import complete_data_reconstruction_pipeline

# Load raw data
publications_df = pd.read_csv('publications.csv')
published_texts_df = pd.read_csv('published_texts.csv')

# Extract and align translations from OCR
results = complete_data_reconstruction_pipeline(
    publications_df,
    published_texts_df,
    sample_size=0,  # 0 = full dataset, 50 = testing
    output_dir='./data/reconstructed'
)

# Get expanded training data (2-3x more examples)
train_df = results['final_training_df']
print(f"✓ Training data: {len(train_df)} pairs")

# ============================================
# STEP 2: Model Training (2-10 hours)
# ============================================
from notebook import T5TranslationModel

# Initialize model
t5_model = T5TranslationModel(model_name='t5-small', device='cuda')

# Prepare datasets with train/val split
train_dataset, val_dataset = t5_model.prepare_data(
    train_df, 
    test_size=0.15
)

# Train model
history = t5_model.train(
    train_dataset, 
    val_dataset, 
    epochs=5, 
    batch_size=8, 
    lr=3e-5
)
print(f"✓ Training complete. Best BLEU: {max(history['val_bleu']):.4f}")

# ============================================
# STEP 3: Generate Submission
# ============================================
test_df = pd.read_csv('test.csv')

# Generate predictions with beam search
predictions = t5_model.predict(
    test_df['transliteration'].tolist(),
    beam_size=5,
    length_penalty=1.0
)

# Create submission file
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'translation': predictions
})
submission_df.to_csv('submission.csv', index=False)
print(f"✓ Submission saved: {len(submission_df)} predictions")

# ============================================
# Optional: Ensemble for Better Results
# ============================================
from notebook import EnsembleTranslator

# Train multiple models
marian_model = MarianTranslationModel(device='cuda')
custom_model = CustomTransformerTranslator(device='cuda')

# Combine models
ensemble = EnsembleTranslator([t5_model, marian_model, custom_model])
ensemble_predictions = ensemble.predict_with_voting(
    test_df['transliteration'].tolist()
)

# Better submission
submission_df['translation'] = ensemble_predictions
submission_df.to_csv('ensemble_submission.csv', index=False)
print(f"✓ Ensemble submission saved")
```

## 🎯 Workflow Recommendations

### For Quick Testing (2-3 hours):
1. Run extraction with `sample_size=50` (10-20 min)
2. Train T5-small for 3 epochs (1-2 hours)
3. Generate predictions (5-10 min)
4. **Expected BLEU**: 0.20-0.30

### For Competitive Results (12-24 hours):
1. Run full extraction `sample_size=0` (8-12 hours)
2. Train T5-base for 5 epochs (6-10 hours)
3. Train MarianMT as second model (4-6 hours)
4. Ensemble predictions (30 min)
5. **Expected BLEU**: 0.35-0.45

### For State-of-the-Art (48+ hours):
1. Full extraction with all 900 PDFs (8-12 hours)
2. Train T5-large (20-30 hours, requires GPU)
3. Train custom transformer with morphological attention (15-20 hours)
4. Train MarianMT baseline (4-6 hours)
5. Ensemble with weighted voting + reranking (1-2 hours)
6. **Expected BLEU**: 0.45-0.55+

## 💡 Tips for Best Results

### Data Extraction Tips:
- **Start Small**: Test with `sample_size=50` before running full dataset
- **Check Intermediate Results**: Inspect `extracted_df` for translation quality
- **Language Distribution**: Monitor how many translations are in each source language
- **Caching**: Save intermediate results to avoid re-processing if pipeline fails
- **Memory Management**: Use chunked processing if you have <16GB RAM

### Model Selection:
- **T5-small**: Fast iteration, good for prototyping (2-4 hours training)
- **T5-base**: Better quality, 3x slower (6-10 hours training)
- **T5-large**: Best quality, requires significant GPU memory

### Hyperparameter Tuning:
- **Beam size**: 4-8 (higher = more diverse but slower)
- **Length penalty**: 0.8-1.2 (controls output length)
- **Learning rate**: 1e-5 to 5e-5 (lower for larger models)
- **Batch size**: Maximize based on GPU memory

### Data Strategies:
- Use validation split to prevent overfitting (10-15%)
- Apply data augmentation sparingly (increases training time)
- Consider curriculum learning for better convergence

### Ensemble Tips:
- Train multiple checkpoints with different seeds
- Combine T5 with MarianMT for diversity
- Use BLEU-based reranking for final selection

## 🔍 Challenges & Solutions

### Challenge 1: Morphological Complexity
**Problem**: Single Akkadian words encode multiple English words
**Solution**: Subword tokenization (BPE) + morphological attention layer

### Challenge 2: Low-Resource Data
**Problem**: Only 1,200 training examples
**Solution**: Transfer learning from pre-trained models + data augmentation

### Challenge 3: Domain Specificity
**Problem**: Ancient administrative texts have unique vocabulary
**Solution**: Leverage lexicon for vocabulary enrichment

### Challenge 4: Variable Length Translations
**Problem**: English uses 40%+ more words than Akkadian
**Solution**: Dynamic length penalty in beam search

## 📊 Expected Performance

### Baseline (No Training):
- BLEU: ~0.05-0.10 (pre-trained model without fine-tuning)

### After Fine-tuning:
- **T5-small**: BLEU ~0.25-0.35
- **T5-base**: BLEU ~0.35-0.45
- **Ensemble**: BLEU ~0.40-0.50

### State-of-the-Art (with extensive tuning):
- BLEU >0.50 (requires extensive experimentation)

## 📚 References & Resources

### Pre-trained Models:
- T5: https://huggingface.co/t5-small
- MarianMT: https://huggingface.co/Helsinki-NLP/opus-mt-mul-en
- mT5: https://huggingface.co/google/mt5-small

### Papers:
- "Exploring the Limits of Transfer Learning with T5" (Raffel et al., 2020)
- "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
- "Attention Is All You Need" (Vaswani et al., 2017)

### Akkadian Resources:
- Electronic Babylonian Library (eBL): https://www.ebl.lmu.de/
- ORACC (Open Richly Annotated Cuneiform Corpus)

## 🎓 Learning Outcomes

This project demonstrates:
1. **Low-resource NMT**: Techniques for translating under-resourced languages
2. **Transfer Learning**: Leveraging pre-trained models for new domains
3. **Morphological Processing**: Handling complex word structures
4. **Ensemble Methods**: Combining models for robust predictions
5. **Ancient Language Processing**: Applying modern NLP to historical texts

## 🔮 Future Work

- **Multilingual Models**: Train on multiple ancient languages simultaneously
- **Back-translation**: Generate synthetic Akkadian from English
- **Contextual Embeddings**: Incorporate document-level context
- **Active Learning**: Identify most valuable examples for annotation
- **Interpretability**: Analyze what the model learns about Akkadian grammar

## 👨‍💻 Author

**Amit Kumar Gope**
- GitHub: [amitkumargope](https://github.com/amitkumargope)
- Project: Deep Past Initiative - Machine Translation Challenge

## 📄 License

This project is part of the Deep Past Challenge competition. Please refer to the competition terms for usage rights.

---

**Last Updated**: December 2025

**Note**: This is a challenging task at the intersection of NLP and Digital Humanities. Don't be discouraged by initial results - low-resource NMT is an active research area!
