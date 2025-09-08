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

# --- Add necessary NLTK imports and setup ---
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

# --- Internal preprocessing function (currently unused but kept for potential future use) ---
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
    """Creates an interactive bar chart of sentiment counts with improved aesthetics."""
    sentiment_counts = df['predicted_label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    sentiment_counts['Sentiment'] = sentiment_counts['Sentiment'].map({1: 'Happy', 0: 'Not Happy', -1: 'Error'})
    
    fig = px.bar(sentiment_counts, 
                 x='Sentiment', 
                 y='Count',
                 color='Sentiment', 
                 color_discrete_map={'Happy': 'mediumseagreen', 'Not Happy': 'indianred', 'Error': 'lightgrey'},
                 title="Sentiment Distribution of Reviews",
                 text_auto=True,  # Display count on bars
                 template='plotly_white'
                )
    
    fig.update_layout(
        title_x=0.5, # Center the title
        xaxis_title=None,
        yaxis_title="Number of Reviews",
        showlegend=False,
        bargap=0.2
    )
    
    fig.update_traces(
        marker_line_width=1.5, 
        marker_line_color="black",
        opacity=0.8,
        marker_cornerradius=8
    )
    return fig

# --- create_common_words_plot function (currently unused) ---
# def create_common_words_plot(df, sentiment_label, n=15):
#     """Creates an interactive bar chart of the most common words for a given sentiment."""
#     # ... (function logic)


def create_time_series_plot(df):
    """Creates an interactive time-series plot of sentiment counts per day with improved aesthetics."""
    if df.empty or 'timestamp' not in df.columns or df['timestamp'].isnull().all():
        return None

    df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
    df.dropna(subset=['date'], inplace=True)

    if df.empty:
        return None

    df['Sentiment'] = df['predicted_label'].map({1: 'Happy', 0: 'Not Happy'})
    daily_counts = df.groupby(['date', 'Sentiment']).size().reset_index(name='Count')

    fig = px.line(daily_counts, 
                  x='date', 
                  y='Count', 
                  color='Sentiment',
                  title="Daily Sentiment Trend",
                  labels={'date': 'Date', 'Count': 'Number of Reviews'},
                  color_discrete_map={'Happy': 'mediumseagreen', 'Not Happy': 'indianred'},
                  markers=True,
                  template='plotly_white'
                 )
    
    fig.update_layout(
        title_x=0.5, # Center the title
        xaxis_title="Date", 
        yaxis_title="Number of Reviews",
        legend_title_text=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True), type="date") # Add a range slider
    )
    
    fig.update_traces(line=dict(width=2.5))
    
    return fig