import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import json
from text_summarization_comparison import TextSummarizationComparison
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure matplotlib for better display
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SummarizationDashboard:
    """Interactive dashboard for comparing text summarization approaches"""
    
    def __init__(self):
        self.comparator = TextSummarizationComparison()
        self.current_results = None
    
    def run_comparison(self, article_url, num_sentences, max_words):
        """Run the comparison with given parameters"""
        try:
            # Fetch article
            article_text = self.comparator.fetch_article(article_url if article_url else None)
            
            # Generate summaries
            deterministic_result = self.comparator.deterministic_extractive_summarization(
                article_text, 
                num_sentences=int(num_sentences)
            )
            
            probabilistic_result = self.comparator.probabilistic_abstractive_summarization(
                article_text,
                max_length=int(max_words)
            )
            
            # Compare summaries
            comparison = self.comparator.compare_summaries(
                article_text,
                deterministic_result['summary'],
                probabilistic_result['summary']
            )
            
            # Store results
            self.current_results = {
                'article_text': article_text,
                'deterministic': deterministic_result,
                'probabilistic': probabilistic_result,
                'comparison': comparison
            }
            
            return (
                deterministic_result['summary'],
                probabilistic_result['summary'],
                self.create_comparison_plot(comparison),
                self.create_metrics_table(comparison),
                article_text[:1000] + "..."  # Show preview of original
            )
            
        except Exception as e:
            logging.error(f"Error in comparison: {e}")
            return (
                f"Error: {str(e)}",
                f"Error: {str(e)}",
                None,
                None,
                f"Error fetching article: {str(e)}"
            )
    
    def create_comparison_plot(self, comparison):
        """Create visualization comparing the two approaches"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Similarity scores
        ax = axes[0, 0]
        similarities = comparison['similarity_scores']
        x = ['Deterministic\nto Original', 'Probabilistic\nto Original', 'Between\nSummaries']
        y = [similarities['summary1_to_original'], 
             similarities['summary2_to_original'], 
             similarities['between_summaries']]
        bars = ax.bar(x, y, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Similarity Scores')
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom')
        
        # Compression ratios
        ax = axes[0, 1]
        compressions = comparison['compression_ratios']
        methods = ['Deterministic', 'Probabilistic']
        ratios = [compressions['summary1'], compressions['summary2']]
        bars = ax.bar(methods, ratios, color=['#1f77b4', '#ff7f0e'])
        ax.set_ylabel('Compression Ratio')
        ax.set_title('Summary Compression')
        ax.set_ylim(0, max(ratios) * 1.2)
        for bar, val in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                   f'{val:.2%}', ha='center', va='bottom')
        
        # Word counts
        ax = axes[1, 0]
        lengths = comparison['lengths']
        categories = ['Original', 'Deterministic', 'Probabilistic']
        values = [lengths['original_words'], 
                 lengths['summary1_words'], 
                 lengths['summary2_words']]
        bars = ax.bar(categories, values, color=['#9467bd', '#1f77b4', '#ff7f0e'])
        ax.set_ylabel('Word Count')
        ax.set_title('Document Lengths')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                   f'{val}', ha='center', va='bottom')
        
        # Readability
        ax = axes[1, 1]
        readability = comparison['readability']
        methods = ['Deterministic', 'Probabilistic']
        avg_lengths = [readability['summary1_avg_sentence_length'], 
                      readability['summary2_avg_sentence_length']]
        bars = ax.bar(methods, avg_lengths, color=['#1f77b4', '#ff7f0e'])
        ax.set_ylabel('Average Words per Sentence')
        ax.set_title('Readability (Sentence Length)')
        for bar, val in zip(bars, avg_lengths):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{val:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_metrics_table(self, comparison):
        """Create a detailed metrics comparison table"""
        metrics_data = {
            'Metric': [
                'Similarity to Original',
                'Compression Ratio',
                'Word Count',
                'Avg Sentence Length',
                'Summary Type'
            ],
            'Deterministic': [
                f"{comparison['similarity_scores']['summary1_to_original']:.3f}",
                f"{comparison['compression_ratios']['summary1']:.1%}",
                f"{comparison['lengths']['summary1_words']}",
                f"{comparison['readability']['summary1_avg_sentence_length']:.1f}",
                "Extractive"
            ],
            'Probabilistic': [
                f"{comparison['similarity_scores']['summary2_to_original']:.3f}",
                f"{comparison['compression_ratios']['summary2']:.1%}",
                f"{comparison['lengths']['summary2_words']}",
                f"{comparison['readability']['summary2_avg_sentence_length']:.1f}",
                "Abstractive"
            ]
        }
        
        df = pd.DataFrame(metrics_data)
        return df
    
    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(title="Text Summarization: Deterministic vs Probabilistic") as interface:
            gr.Markdown("""
            # Text Summarization Comparison: Deterministic vs Probabilistic
            
            This tool compares two approaches to summarizing Tim O'Reilly's "The End of Programming":
            - **Deterministic (Extractive)**: Uses TF-IDF to select the most important sentences
            - **Probabilistic (Abstractive)**: Uses OpenAI GPT-4 to generate new summary text
            """)
            
            with gr.Row():
                with gr.Column():
                    article_url = gr.Textbox(
                        label="Article URL (leave empty for default)",
                        placeholder="https://www.oreilly.com/radar/the-end-of-programming/",
                        value=""
                    )
                    num_sentences = gr.Slider(
                        minimum=3, maximum=10, value=5, step=1,
                        label="Number of sentences for extractive summary"
                    )
                    max_words = gr.Slider(
                        minimum=50, maximum=300, value=150, step=25,
                        label="Target words for abstractive summary"
                    )
                    compare_btn = gr.Button("Run Comparison", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Deterministic Summary (Extractive)")
                    deterministic_output = gr.Textbox(
                        label="Selected sentences from original",
                        lines=8
                    )
                
                with gr.Column():
                    gr.Markdown("### Probabilistic Summary (Abstractive)")
                    probabilistic_output = gr.Textbox(
                        label="AI-generated summary",
                        lines=8
                    )
            
            with gr.Row():
                comparison_plot = gr.Plot(label="Visual Comparison")
            
            with gr.Row():
                metrics_table = gr.Dataframe(
                    label="Detailed Metrics Comparison",
                    headers=["Metric", "Deterministic", "Probabilistic"]
                )
            
            with gr.Row():
                with gr.Accordion("Original Article Preview", open=False):
                    original_preview = gr.Textbox(
                        label="First 1000 characters",
                        lines=10
                    )
            
            # Connect the comparison function
            compare_btn.click(
                fn=self.run_comparison,
                inputs=[article_url, num_sentences, max_words],
                outputs=[
                    deterministic_output,
                    probabilistic_output,
                    comparison_plot,
                    metrics_table,
                    original_preview
                ]
            )
            
            gr.Markdown("""
            ## Key Differences:
            
            | Aspect | Deterministic (Extractive) | Probabilistic (Abstractive) |
            |--------|---------------------------|----------------------------|
            | **Method** | Selects existing sentences | Generates new text |
            | **Consistency** | Same input → Same output | Same input → Different outputs |
            | **Creativity** | None (verbatim extraction) | High (paraphrasing/synthesis) |
            | **Factual Accuracy** | 100% (original text) | May hallucinate or misinterpret |
            | **Fluency** | May lack coherence | Usually more coherent |
            | **Speed** | Very fast | Slower (API calls) |
            | **Cost** | Free (local computation) | API costs |
            """)
        
        return interface


if __name__ == "__main__":
    dashboard = SummarizationDashboard()
    interface = dashboard.create_interface()
    interface.launch()