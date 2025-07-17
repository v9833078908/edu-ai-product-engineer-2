import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Tuple
from openai import OpenAI
import os
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('summarization_comparison.log'),
        logging.StreamHandler()
    ]
)


class TextSummarizationComparison:
    """Compare deterministic vs probabilistic text summarization approaches"""
    
    def __init__(self):
        """Initialize with necessary NLTK downloads and API setup"""
        # Download required NLTK data
        for dataset in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
            try:
                nltk.data.find(f'tokenizers/{dataset}')
            except LookupError:
                nltk.download(dataset)
        
        # Initialize cache for API responses
        self.cache_file = 'summarization_cache.json'
        self.cache = self.load_cache()
        
        # Setup OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4-turbo-preview"
        else:
            logging.warning("No OPENAI_API_KEY found. Probabilistic summarization will not work.")
            self.client = None
            self.model = None
    
    def load_cache(self) -> Dict:
        """Load cache from file"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def fetch_article(self, url: str = None) -> str:
        """Fetch Tim O'Reilly's article"""
        if not url:
            url = "https://www.oreilly.com/radar/the-end-of-programming/"
        
        # Check cache first
        if url in self.cache:
            logging.info(f"Using cached article from {url}")
            return self.cache[url]['content']
        
        logging.info(f"Fetching article from {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract main content
        article_content = ""
        for element in soup.find_all(['p', 'h1', 'h2', 'h3']):
            text = element.get_text().strip()
            if text:
                article_content += text + "\n\n"
        
        # Cache the result
        self.cache[url] = {
            'content': article_content,
            'fetched_at': datetime.now().isoformat()
        }
        self.save_cache()
        
        return article_content
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for summarization"""
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Clean sentences
        clean_sentences = []
        for sentence in sentences:
            # Remove extra whitespace
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            # Remove very short sentences
            if len(sentence.split()) > 5:
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def deterministic_extractive_summarization(self, text: str, num_sentences: int = 5) -> Dict:
        """
        Deterministic approach using TF-IDF and sentence ranking
        This is extractive summarization - selecting existing sentences
        """
        logging.info("Running deterministic extractive summarization")
        
        # Preprocess
        sentences = self.preprocess_text(text)
        
        if len(sentences) <= num_sentences:
            return {
                'method': 'deterministic_extractive',
                'summary': ' '.join(sentences),
                'sentences_selected': len(sentences),
                'total_sentences': len(sentences),
                'scores': {}
            }
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores based on TF-IDF values
        sentence_scores = {}
        for idx, sentence in enumerate(sentences):
            score = tfidf_matrix[idx].sum()
            sentence_scores[idx] = float(score)
        
        # Rank sentences and select top N
        ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        selected_indices = sorted([idx for idx, score in ranked_sentences[:num_sentences]])
        
        # Build summary maintaining original order
        summary = ' '.join([sentences[idx] for idx in selected_indices])
        
        return {
            'method': 'deterministic_extractive',
            'summary': summary,
            'sentences_selected': num_sentences,
            'total_sentences': len(sentences),
            'scores': {sentences[idx]: score for idx, score in ranked_sentences[:num_sentences]}
        }
    
    def probabilistic_abstractive_summarization(self, text: str, max_length: int = 150) -> Dict:
        """
        Probabilistic approach using LLM for abstractive summarization
        This generates new text that captures the essence
        """
        if not self.client:
            return {
                'method': 'probabilistic_abstractive',
                'summary': 'Error: No OpenAI API key configured',
                'error': True
            }
        
        logging.info("Running probabilistic abstractive summarization")
        
        # Check cache for LLM response
        cache_key = f"llm_summary_{hash(text[:100])}"
        if cache_key in self.cache:
            logging.info("Using cached LLM summary")
            return self.cache[cache_key]
        
        prompt = f"""
        Please provide a concise abstractive summary of the following article about "The End of Programming" by Tim O'Reilly.
        The summary should:
        1. Be approximately {max_length} words
        2. Capture the main arguments and key insights
        3. Use your own words to synthesize the ideas
        4. Focus on the implications for the future of programming
        
        Article:
        {text[:3000]}  # Limiting to avoid token limits
        
        Summary:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, insightful summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=max_length * 2  # Approximate tokens needed
            )
            
            summary = response.choices[0].message.content.strip()
            
            result = {
                'method': 'probabilistic_abstractive',
                'summary': summary,
                'word_count': len(summary.split()),
                'model': self.model,
                'temperature': 0.7
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.save_cache()
            
            return result
            
        except Exception as e:
            logging.error(f"Error in LLM summarization: {e}")
            return {
                'method': 'probabilistic_abstractive',
                'summary': f'Error generating summary: {str(e)}',
                'error': True
            }
    
    def compare_summaries(self, original_text: str, summary1: str, summary2: str) -> Dict:
        """Compare two summaries using various metrics"""
        logging.info("Comparing summaries")
        
        # Calculate similarity to original
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([original_text, summary1, summary2])
        
        sim1_to_original = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        sim2_to_original = cosine_similarity(vectors[0:1], vectors[2:3])[0][0]
        sim_between = cosine_similarity(vectors[1:2], vectors[2:3])[0][0]
        
        # Calculate compression ratios
        compression1 = len(summary1.split()) / len(original_text.split())
        compression2 = len(summary2.split()) / len(original_text.split())
        
        # Calculate readability (simple metric based on sentence length)
        def avg_sentence_length(text):
            sentences = nltk.sent_tokenize(text)
            if not sentences:
                return 0
            return np.mean([len(s.split()) for s in sentences])
        
        readability1 = avg_sentence_length(summary1)
        readability2 = avg_sentence_length(summary2)
        
        return {
            'similarity_scores': {
                'summary1_to_original': float(sim1_to_original),
                'summary2_to_original': float(sim2_to_original),
                'between_summaries': float(sim_between)
            },
            'compression_ratios': {
                'summary1': float(compression1),
                'summary2': float(compression2)
            },
            'readability': {
                'summary1_avg_sentence_length': float(readability1),
                'summary2_avg_sentence_length': float(readability2)
            },
            'lengths': {
                'original_words': len(original_text.split()),
                'summary1_words': len(summary1.split()),
                'summary2_words': len(summary2.split())
            }
        }
    
    def run_comparison(self, article_url: str = None) -> Dict:
        """Run full comparison between deterministic and probabilistic approaches"""
        # Fetch article
        article_text = self.fetch_article(article_url)
        logging.info(f"Article fetched: {len(article_text)} characters")
        
        # Generate summaries
        deterministic_result = self.deterministic_extractive_summarization(article_text)
        probabilistic_result = self.probabilistic_abstractive_summarization(article_text)
        
        # Compare summaries
        comparison = self.compare_summaries(
            article_text,
            deterministic_result['summary'],
            probabilistic_result['summary']
        )
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'article_length': len(article_text),
            'deterministic': deterministic_result,
            'probabilistic': probabilistic_result,
            'comparison': comparison
        }
        
        # Save results
        with open('summarization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info("Comparison complete. Results saved to summarization_results.json")
        
        return results


if __name__ == "__main__":
    # Run the comparison
    comparator = TextSummarizationComparison()
    results = comparator.run_comparison()
    
    # Print summary of results
    print("\n=== DETERMINISTIC (EXTRACTIVE) SUMMARY ===")
    print(results['deterministic']['summary'][:500] + "...")
    
    print("\n=== PROBABILISTIC (ABSTRACTIVE) SUMMARY ===")
    print(results['probabilistic']['summary'][:500] + "...")
    
    print("\n=== COMPARISON METRICS ===")
    print(json.dumps(results['comparison'], indent=2))