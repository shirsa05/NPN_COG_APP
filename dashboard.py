import pandas as pd
import plotly.express as px
from collections import Counter
from predictor import preprocess_text # Import the preprocessing function

def create_sentiment_distribution_plot(df):
    """Creates an interactive bar chart of sentiment counts."""
    sentiment_counts = df['predicted_label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    sentiment_counts['Sentiment'] = sentiment_counts['Sentiment'].map({1: 'Happy', 0: 'Not Happy'})
    fig = px.bar(sentiment_counts, x='Sentiment', y='Count',
                   color='Sentiment', color_discrete_map={'Happy': 'green', 'Not Happy': 'red'},
                   title="Count of Happy vs. Not Happy Reviews")
    return fig

def create_common_words_plot(df, sentiment_label, lemmatizer, stop_words, n=15):
    """Creates an interactive bar chart of the most common words for a given sentiment."""
    sentiment_map = {1: 'Happy', 0: 'Not Happy'}
    color_map = {1: 'green', 0: 'red'}
    
    corpus = df[df['predicted_label'] == sentiment_label]['Description']
    all_text = " ".join(str(text) for text in corpus)
    processed_text = preprocess_text(all_text, lemmatizer, stop_words)
    
    words = processed_text.split()
    word_counts = Counter(words)
    top_words = pd.DataFrame(word_counts.most_common(n), columns=['Word', 'Count'])
    
    fig = px.bar(top_words, x='Count', y='Word', orientation='h',
                   title=f"Top Words in {sentiment_map[sentiment_label]} Reviews",
                   color_discrete_sequence=[color_map[sentiment_label]])
    return fig


def create_time_series_plot(df):
    """Creates an interactive time-series plot of sentiment counts per day."""
    if df.empty or 'timestamp' not in df.columns or df['timestamp'].isnull().all():
        return None

    # Ensure timestamp is in datetime format, handling potential errors
    df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
    df.dropna(subset=['date'], inplace=True)

    if df.empty:
        return None

    # Map labels to human-readable names for the plot legend
    df['Sentiment'] = df['predicted_label'].map({1: 'Happy', 0: 'Not Happy'})

    # Group by date and sentiment to get daily counts
    daily_counts = df.groupby(['date', 'Sentiment']).size().reset_index(name='Count')

    # Create an interactive line chart with markers
    fig = px.line(daily_counts,
                  x='date',
                  y='Count',
                  color='Sentiment',
                  title="Daily Sentiment Trend (All Reviews)",
                  labels={'date': 'Date', 'Count': 'Number of Reviews'},
                  color_discrete_map={'Happy': 'mediumseagreen', 'Not Happy': 'indianred'},
                  markers=True) # Add markers to data points

    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Reviews")
    return fig