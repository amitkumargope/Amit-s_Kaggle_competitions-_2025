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

## 🚀 Quick Start

### 1. Setup Environment
```python
# All dependencies are pre-installed in the notebook
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 2. Load Data
```python
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
```

### 3. Train Model
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

### 4. Generate Predictions
```python
submission = generate_submission(t5_model, test_df, 'submission.csv')
```

## 💡 Tips for Best Results

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
