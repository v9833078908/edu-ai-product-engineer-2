# Text Summarization Comparison: Deterministic vs Probabilistic

This system compares deterministic (extractive) and probabilistic (abstractive) approaches to text summarization using Tim O'Reilly's article "The End of Programming" as a case study.

## Overview

### Deterministic/Extractive Approach
- **Method**: TF-IDF based sentence ranking
- **Process**: Selects existing sentences from the original text
- **Output**: Always the same for given input
- **Pros**: Factually accurate, fast, no hallucination
- **Cons**: May lack coherence, limited creativity

### Probabilistic/Abstractive Approach
- **Method**: LLM-based text generation (OpenAI GPT-4)
- **Process**: Generates new text that captures the essence
- **Output**: Can vary between runs
- **Pros**: More coherent, creative synthesis
- **Cons**: May hallucinate, requires API, slower

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key for LLM summarization:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Option 1: Interactive Dashboard
```bash
python summarization_dashboard.py
```
This launches a Gradio web interface where you can:
- Adjust summarization parameters
- See visual comparisons
- Compare metrics side-by-side

### Option 2: Command Line
```bash
python run_comparison.py
```
This runs a single comparison and displays results in the terminal.

### Option 3: Python API
```python
from text_summarization_comparison import TextSummarizationComparison

comparator = TextSummarizationComparison()
results = comparator.run_comparison()
```

## Key Features

1. **Automatic Article Fetching**: Downloads and parses the article from the web
2. **Caching**: Saves API responses to avoid redundant calls
3. **Comprehensive Metrics**:
   - Cosine similarity to original
   - Compression ratios
   - Word counts
   - Readability scores
4. **Visualization**: Interactive plots comparing both approaches
5. **Logging**: Detailed logs saved to `summarization_comparison.log`

## Files

- `text_summarization_comparison.py`: Core comparison logic
- `summarization_dashboard.py`: Gradio web interface
- `run_comparison.py`: Simple CLI runner
- `summarization_cache.json`: Cached API responses
- `summarization_results.json`: Latest comparison results
- `summarization_comparison.log`: Detailed execution logs

## Example Results

The system reveals interesting differences:
- Deterministic summaries maintain exact phrasing but may jump between topics
- Probabilistic summaries create smoother narratives but may miss specific details
- Similarity scores show how each approach relates to the original
- Compression ratios demonstrate efficiency of each method