from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

def perform_topic_modeling(texts, n_topics=5):
    """Perform LDA topic modeling on text data."""
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    # Extract topics
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics.append(top_words)
    return topics, lda, vectorizer

def analyze_topics():
    """Analyze topics and save results."""
    df = pd.read_csv('data/processed/combined_data.csv')
    texts = df['clean_text'].tolist()
    topics, lda, vectorizer = perform_topic_modeling(texts)
    
    # Save topics to file
    with open('data/processed/topics.txt', 'w') as f:
        for idx, topic in enumerate(topics):
            f.write(f"Topic {idx + 1}: {', '.join(topic)}\n")
    return topics

if __name__ == "__main__":
    analyze_topics()