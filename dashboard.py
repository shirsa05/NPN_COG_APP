import pandas as pd
import plotly.express as px
from collections import Counter
import re
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- NEW: Add necessary NLTK imports and setup ---
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/punkt_tab')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

# --- NEW: Internal preprocessing function ---
# def _preprocess_for_word_count(text):
#     """A private preprocessing function for the dashboard's use."""
#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words('english'))
    
#     text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
#     text = text.lower()
#     tokens = word_tokenize(text)
#     processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
#     return " ".join(processed_tokens)


def create_sentiment_distribution_plot(df):
    """Creates an interactive bar chart of sentiment counts."""
    # This function remains unchanged
    sentiment_counts = df['predicted_label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    sentiment_counts['Sentiment'] = sentiment_counts['Sentiment'].map({1: 'Happy', 0: 'Not Happy', -1: 'Error'})
    fig = px.bar(sentiment_counts, x='Sentiment', y='Count',
                   color='Sentiment', color_discrete_map={'Happy': 'green', 'Not Happy': 'red', 'Error': 'grey'},
                   title="Count of Happy vs. Not Happy Reviews")
    return fig

# # --- UPDATED: create_common_words_plot function ---
# def create_common_words_plot(df, sentiment_label, n=15):
#     """Creates an interactive bar chart of the most common words for a given sentiment."""
#     sentiment_map = {1: 'Happy', 0: 'Not Happy'}
#     color_map = {1: 'green', 0: 'red'}
    
#     corpus = df[df['predicted_label'] == sentiment_label]['Description']
#     if corpus.empty:
#         return px.bar(title=f"No '{sentiment_map[sentiment_label]}' reviews to analyze")

#     # Use the new internal preprocessing function
#     all_text = " ".join(str(text) for text in corpus)
#     processed_text = _preprocess_for_word_count(all_text)
    
#     words = processed_text.split()
#     if not words:
#         return px.bar(title=f"No common words found for '{sentiment_map[sentiment_label]}' reviews")

#     word_counts = Counter(words)
#     top_words = pd.DataFrame(word_counts.most_common(n), columns=['Word', 'Count'])
    
#     fig = px.bar(top_words, x='Count', y='Word', orientation='h',
#                    title=f"Top Words in {sentiment_map[sentiment_label]} Reviews",
#                    color_discrete_sequence=[color_map[sentiment_label]])
#     fig.update_yaxes(autorange="reversed")
#     return fig


def create_time_series_plot(df):
    """Creates an interactive time-series plot of sentiment counts per day."""
    # This function remains unchanged
    if df.empty or 'timestamp' not in df.columns or df['timestamp'].isnull().all():
        return None

    df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
    df.dropna(subset=['date'], inplace=True)

    if df.empty:
        return None

    df['Sentiment'] = df['predicted_label'].map({1: 'Happy', 0: 'Not Happy'})
    daily_counts = df.groupby(['date', 'Sentiment']).size().reset_index(name='Count')

    fig = px.line(daily_counts, x='date', y='Count', color='Sentiment',
                  title="Daily Sentiment Trend (All Reviews)",
                  labels={'date': 'Date', 'Count': 'Number of Reviews'},
                  color_discrete_map={'Happy': 'mediumseagreen', 'Not Happy': 'indianred'},
                  markers=True)
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Reviews")
    return fig
