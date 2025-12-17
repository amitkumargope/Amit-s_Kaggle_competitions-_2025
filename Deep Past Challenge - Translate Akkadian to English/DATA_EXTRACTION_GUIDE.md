# Data Extraction and Reconstruction Pipeline

## Overview

This document describes the **critical preprocessing step** for the Akkadian-English translation challenge: extracting and aligning training data from OCR'd scholarly publications.

## The Challenge

The competition provides:
- ✅ **train.csv**: 1,200+ pre-aligned text pairs  
- ❌ **publications.csv**: 554MB of OCR'd PDFs with translations (NOT aligned!)
- ❌ **published_texts.csv**: 11,000+ transliterations (NO translations!)

**The problem**: Before training any neural model, you must reconstruct the parallel corpus by extracting translations from the publications and matching them with transliterations.

## Pipeline Architecture

```
publications.csv (OCR text)  +  published_texts.csv (transliterations)
                     ↓
         [1. Translation Extraction]
                     ↓
         [2. Language Normalization]
                     ↓
         [3. Sentence Alignment]
                     ↓
    Parallel Training Corpus (Akkadian ↔ English)
```

## Step 1: Translation Extraction

### Input
- **publications.csv**: 900 PDFs, ~553MB of OCR text
- Multilingual scholarly publications (English, German, French, Turkish)
- Various citation formats (ICK, BIN, CCT, KTS, ATHE, etc.)

### Process

#### 1.1 Document Matching
Match transliterations with their translations using:
- **Catalog identifiers**: ICK 1 146, BIN VI 39, CCT II 6, etc.
- **Museum numbers**: BM 113610, P361099, etc.
- **Text labels**: Cuneiform Tablet designations

#### 1.2 Translation Extraction
Use pattern matching to extract translations:
```python
# English quotes
r'"([^"]{10,500})"'

# German quotes  
r'„([^"]{10,500})"'

# French guillemets
r'«([^»]{10,500})»'
```

#### 1.3 Context Analysis
Extract surrounding context (±250 chars) to capture:
- Transliteration-translation pairs
- Commentary and annotations
- Multiple translation variants

### Challenges
- **OCR errors**: Misrecognized characters (especially Akkadian with diacritics)
- **Mixed layouts**: Multi-column text, footnotes, headers
- **Citation variations**: Same tablet referenced multiple ways
- **Partial texts**: Fragments and damaged sections marked with gaps

### Output
```python
{
    'oare_id': '004a7dbd-57ce-46f8-9691-409be61c676e',
    'label': 'ICK I 49a',
    'transliteration': 'KIŠIB ma-nu-ba-lúm-a-šur...',
    'translation': 'Seal of Mannum-balum-Aššur son of...',
    'source': 'ocr_extraction'
}
```

## Step 2: Language Normalization

### Input
Extracted translations in multiple languages:
- **German**: ~40% of publications (Assyriological tradition)
- **English**: ~35%
- **French**: ~15%
- **Turkish**: ~10%

### Process

#### 2.1 Language Detection
Heuristic-based detection using character/word indicators:
```python
german_indicators = ['ü', 'ä', 'ö', 'ß', 'und', 'der', 'die']
french_indicators = ['é', 'è', 'à', 'ç', 'et', 'le', 'la']
turkish_indicators = ['ı', 'ş', 'ğ', 've', 'bir']
```

#### 2.2 Neural Translation to English
Use Helsinki-NLP Marian models:
- **German→English**: `opus-mt-de-en`
- **French→English**: `opus-mt-fr-en`
- **Turkish→English**: `opus-mt-tr-en`

Batch processing for efficiency:
```python
normalizer = MultilingualTranslationNormalizer()
normalized_df = normalizer.process_multilingual_dataframe(extracted_df)
```

### Challenges
- **Domain terminology**: Ancient Near Eastern terms may not translate well
- **Scholarly conventions**: "[...]" for gaps, italics for uncertain readings
- **Mixed language**: Some publications use multiple languages
- **Technical vocabulary**: Weights, measures, legal terms

### Output
All translations normalized to English:
```python
{
    'original_lang': 'german',
    'original_text': 'Siegel des Mannum-balum-Aššur, Sohn des...',
    'english_text': 'Seal of Mannum-balum-Aššur, son of...'
}
```

## Step 3: Sentence-Level Alignment

### Input
- Akkadian transliterations (long documents)
- English translations (long documents)
- Goal: Create sentence-level parallel pairs

### Process

#### 3.1 Akkadian Sentence Segmentation
Akkadian doesn't use punctuation. Segment using linguistic markers:

**Boundary indicators**:
- `-ma`: Connective suffix ("and then", "and")
- `um-ma ... -ma`: "Thus said X:" (quote introduction)
- `qí-bí-ma`: "Say!" (letter opening formula)

**Heuristics**:
- Max 50 words per segment
- Split at structural markers
- Preserve semantic units

Example:
```
Input: "um-ma Puzur-Aššur-ma ana Buzazu qíbi-ma..."
       ↓
Segments:
1. "um-ma Puzur-Aššur-ma ana Buzazu qíbi-ma"
2. "aššum DUB ša 45 GÚ URUDU..."
```

#### 3.2 English Sentence Segmentation
Standard sentence splitting:
```python
sentences = re.split(r'[.!?]+\s+', english_text)
```

#### 3.3 Length-Ratio Alignment
Align using cumulative word count ratios:

```python
# Calculate progress through Akkadian
akk_progress = akkadian_words_seen / total_akkadian_words

# Calculate progress through English  
eng_progress = english_words_seen / total_english_words

# Align when progress rates are similar
if abs(akk_progress - eng_progress) < 0.1:
    create_alignment(akk_sentence, eng_sentence)
```

### Challenges
- **Length variability**: 1 Akkadian word = 1-3 English words
- **Structural differences**: English uses more function words
- **Translation styles**: Some scholars translate word-for-word, others freely
- **Fragmentary texts**: Gaps make alignment uncertain

### Output
Sentence-aligned parallel pairs:
```python
{
    'akkadian_sentence': 'um-ma Puzur-Aššur-ma ana Buzazu qíbi-ma',
    'english_sentence': 'Thus Puzur-Aššur, say to Buzazu:',
    'akkadian_words': 7,
    'english_words': 7,
    'length_ratio': 1.0
}
```

## Step 4: Data Augmentation

### Process

#### 4.1 Merge Data Sources
Combine:
- Existing train.csv (1,200 pairs)
- Extracted OCR translations (variable)
- Sentence alignments from both

#### 4.2 Deduplication
Remove duplicates based on Akkadian text:
```python
df = df.drop_duplicates(subset=['akkadian_sentence'], keep='first')
```

#### 4.3 Quality Filtering
Filter low-quality pairs:
- Min length: 10 characters
- Max length: 500 characters
- Length ratio: 0.5 to 3.0
- Remove incomplete OCR extractions

#### 4.4 Statistics
Calculate and report:
- Total sentence pairs
- Average lengths
- Length ratio distribution
- Source distribution

### Output
Enhanced training corpus:
```
Original:     1,200 document pairs → ~3,000 sentence pairs
+ Extraction: 100 document pairs → ~500 sentence pairs
= Final:      ~3,500 high-quality sentence pairs
```

## Implementation Guide

### Quick Start
```python
# 1. Load data
publications_df = pd.read_csv('publications.csv')
published_texts_df = pd.read_csv('published_texts.csv')
train_df = pd.read_csv('train.csv')

# 2. Extract translations
extractor = PublicationTranslationExtractor()
extracted = extractor.process_publications(
    published_texts_df, 
    publications_df, 
    sample_size=100  # Start small
)

# 3. Normalize to English
normalizer = MultilingualTranslationNormalizer()
normalized = normalizer.process_multilingual_dataframe(extracted)

# 4. Align sentences
aligner = SentenceAligner()
aligned = aligner.process_dataframe(normalized)

# 5. Merge and save
final_corpus = pd.concat([train_aligned, aligned])
final_corpus.to_csv('enhanced_training_data.csv')
```

### Performance Tips

#### Start Small, Scale Up
```python
# Initial test: 10 documents (~5 minutes)
extracted = extractor.process_publications(..., sample_size=10)

# Medium run: 100 documents (~1 hour)  
extracted = extractor.process_publications(..., sample_size=100)

# Full extraction: 900 documents (~overnight)
extracted = extractor.process_publications(..., sample_size=None)
```

#### Batch Processing
Process publications in chunks to manage memory:
```python
chunk_size = 100
for i in range(0, len(published_texts_df), chunk_size):
    chunk = published_texts_df[i:i+chunk_size]
    extracted_chunk = extractor.process_publications(chunk, publications_df)
    extracted_chunk.to_csv(f'extracted_chunk_{i}.csv')
```

#### Parallel Processing
Use multiprocessing for OCR text extraction:
```python
from multiprocessing import Pool

def process_text(args):
    text_id, ocr_text = args
    return extractor.extract_translations_from_text(ocr_text)

with Pool(processes=4) as pool:
    results = pool.map(process_text, text_ocr_pairs)
```

## Expected Results

### Coverage
- **Existing training data**: 1,200 documents → ~3,000 sentence pairs
- **PDF extraction (100 docs)**: ~500 additional sentence pairs
- **PDF extraction (all 900)**: ~2,000-5,000 additional sentence pairs

### Quality Metrics
- **Alignment accuracy**: ~80-90% for clear texts
- **OCR quality**: Variable (60-95% depending on PDF quality)
- **Translation accuracy**: ~85% after normalization

### Time Investment
| Task | Sample Size | Time Required |
|------|-------------|---------------|
| Setup | - | 10 minutes |
| Small test | 10 docs | 5 minutes |
| Medium run | 100 docs | 1 hour |
| Full extraction | 900 docs | 8-12 hours |
| Normalization | 500 texts | 2-4 hours |
| Sentence alignment | All | 1-2 hours |

## Common Issues and Solutions

### Issue 1: OCR Quality
**Problem**: Garbled text, missing diacritics
**Solution**: 
- Focus on high-quality PDFs first
- Use fuzzy matching for document IDs
- Manually verify sample of extractions

### Issue 2: Language Detection Errors
**Problem**: Misclassified languages
**Solution**:
- Improve heuristics with more indicators
- Use langdetect library for better accuracy
- Manual correction for ambiguous cases

### Issue 3: Alignment Failures
**Problem**: Sentences don't align properly
**Solution**:
- Adjust length ratio threshold
- Use alternative alignment methods (1:1, by markers)
- Consider document-level alignment for short texts

### Issue 4: Memory Issues
**Problem**: publications.csv is 554MB
**Solution**:
- Process in chunks
- Use iterators instead of loading full DataFrame
- Clear memory after each chunk

## Best Practices

### 1. Incremental Development
Start with known good examples:
```python
# Test on ICK 1 146 (known complete publication)
test_id = 'b05376c2-fc3d-49f8-9792-a25c0df9c383'
test_result = extractor.process_single_text(test_id)
```

### 2. Quality Assurance
Validate each step:
```python
# Check extraction
print(f"Extracted: {len(extracted)}/{len(published_texts_df)}")

# Check normalization
print(f"Languages: {normalized['detected_language'].value_counts()}")

# Check alignment
print(f"Avg ratio: {aligned['length_ratio'].mean():.2f}")
```

### 3. Versioning
Save intermediate results:
```python
extracted.to_csv('v1_extracted.csv')
normalized.to_csv('v1_normalized.csv')
aligned.to_csv('v1_aligned.csv')
```

### 4. Documentation
Log decisions and parameters:
```python
metadata = {
    'extraction_date': datetime.now(),
    'sample_size': 100,
    'alignment_method': 'length_ratio',
    'quality_threshold': 0.8
}
with open('extraction_metadata.json', 'w') as f:
    json.dump(metadata, f)
```

## Conclusion

This data reconstruction pipeline is **essential** for maximizing performance on the Akkadian-English translation task. While the provided train.csv gives a baseline, extracting and aligning the full publication corpus can:

- **Double or triple** your training data
- Provide **more diverse** examples
- Include **longer documents** with context
- Cover **more tablet types** and time periods

The time investment (8-12 hours) is well worth it for the potential performance gains in this low-resource translation challenge.

---

**See notebook sections 2A-2D for complete implementation!**
