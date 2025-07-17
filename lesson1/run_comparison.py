#!/usr/bin/env python3
"""
Simple script to run text summarization comparison
"""

from text_summarization_comparison import TextSummarizationComparison
import json
import sys


def main():
    print("=== Text Summarization Comparison: Deterministic vs Probabilistic ===\n")
    
    # Initialize comparator
    comparator = TextSummarizationComparison()
    
    # Run comparison
    print("Fetching and analyzing Tim O'Reilly's 'The End of Programming'...")
    results = comparator.run_comparison()
    
    # Display results
    print("\n" + "="*80)
    print("DETERMINISTIC (EXTRACTIVE) SUMMARY")
    print("="*80)
    print(f"Method: Selecting {results['deterministic']['sentences_selected']} most important sentences")
    print(f"Total sentences analyzed: {results['deterministic']['total_sentences']}")
    print("\nSummary:")
    print("-"*40)
    print(results['deterministic']['summary'])
    
    print("\n" + "="*80)
    print("PROBABILISTIC (ABSTRACTIVE) SUMMARY")
    print("="*80)
    print(f"Method: AI-generated summary using {results['probabilistic'].get('model', 'LLM')}")
    print(f"Word count: {results['probabilistic'].get('word_count', 'N/A')}")
    print("\nSummary:")
    print("-"*40)
    print(results['probabilistic']['summary'])
    
    print("\n" + "="*80)
    print("COMPARISON METRICS")
    print("="*80)
    
    comparison = results['comparison']
    
    print("\nSimilarity Scores (Cosine Similarity):")
    print(f"  - Deterministic to Original: {comparison['similarity_scores']['summary1_to_original']:.3f}")
    print(f"  - Probabilistic to Original: {comparison['similarity_scores']['summary2_to_original']:.3f}")
    print(f"  - Between Summaries: {comparison['similarity_scores']['between_summaries']:.3f}")
    
    print("\nCompression Ratios:")
    print(f"  - Deterministic: {comparison['compression_ratios']['summary1']:.1%}")
    print(f"  - Probabilistic: {comparison['compression_ratios']['summary2']:.1%}")
    
    print("\nDocument Lengths (words):")
    print(f"  - Original: {comparison['lengths']['original_words']}")
    print(f"  - Deterministic: {comparison['lengths']['summary1_words']}")
    print(f"  - Probabilistic: {comparison['lengths']['summary2_words']}")
    
    print("\nReadability (avg words per sentence):")
    print(f"  - Deterministic: {comparison['readability']['summary1_avg_sentence_length']:.1f}")
    print(f"  - Probabilistic: {comparison['readability']['summary2_avg_sentence_length']:.1f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Analyze results
    if comparison['similarity_scores']['summary1_to_original'] > comparison['similarity_scores']['summary2_to_original']:
        print("✓ Deterministic summary is more similar to the original text")
    else:
        print("✓ Probabilistic summary captures the essence while using different wording")
    
    if comparison['compression_ratios']['summary1'] < comparison['compression_ratios']['summary2']:
        print("✓ Deterministic summary achieves better compression")
    else:
        print("✓ Probabilistic summary is more concise")
    
    if comparison['similarity_scores']['between_summaries'] < 0.5:
        print("✓ The two approaches produce significantly different summaries")
    else:
        print("✓ Both approaches capture similar content despite different methods")
    
    print("\nFull results saved to: summarization_results.json")
    print("Logs saved to: summarization_comparison.log")


if __name__ == "__main__":
    main()