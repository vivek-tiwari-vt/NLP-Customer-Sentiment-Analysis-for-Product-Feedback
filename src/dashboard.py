import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def run_dashboard():
    """Run Streamlit dashboard for sentiment analysis."""
    st.title("Customer Sentiment Analysis Dashboard")
    
    # Load data
    df = pd.read_csv('data/processed/combined_data.csv')
    
    # Sentiment Distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['target'].value_counts().rename({0: 'Negative', 1: 'Positive'})
    st.bar_chart(sentiment_counts)
    
    # Placeholder for trends (assuming date column exists or can be mocked)
    st.subheader("Sentiment Trends Over Time")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['target'], label='Sentiment Score')
    ax.set_xlabel('Review Index')
    ax.set_ylabel('Sentiment (0=Negative, 1=Positive)')
    st.pyplot(fig)
    
    # Topics
    st.subheader("Top Topics Driving Sentiment")
    with open('data/processed/topics.txt', 'r') as f:
        topics = f.read()
    st.text(topics)

if __name__ == "__main__":
    run_dashboard()