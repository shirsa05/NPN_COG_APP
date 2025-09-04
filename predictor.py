import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import ssl

# --- NLTK Setup ---
def setup_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    packages = ['stopwords', 'punkt', 'wordnet', 'punkt_tab']
    for pkg in packages:
        try:
            nltk.data.find(f'corpora/{pkg}' if pkg in ['stopwords', 'wordnet'] else f'tokenizers/{pkg}')
        except LookupError:
            nltk.download(pkg)

setup_nltk()

# --- Load Assets ---
def load_assets():
    """Loads all necessary ML assets."""
    model = joblib.load('models/new_sentiment_model.joblib')
    vectorizer = joblib.load('models/new_tfidf_vectorizer.joblib')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return model, vectorizer, lemmatizer, stop_words

# --- Preprocessing and Prediction ---
def preprocess_text(text, lemmatizer, stop_words):
    """Cleans and preprocesses a single piece of text using NLTK."""
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(processed_tokens)

def predict_sentiment(new_review, vectorizer, model, lemmatizer, stop_words):
    """Predicts the sentiment of a new, raw text review."""
    processed_review = preprocess_text(new_review, lemmatizer, stop_words)
    review_vector = vectorizer.transform([processed_review])
    prediction = model.predict(review_vector)
    prediction_proba = model.predict_proba(review_vector)

    if prediction[0] == 1:
        sentiment_label = 1
        confidence = prediction_proba[0][1]
    else:
        sentiment_label = 0
        confidence = prediction_proba[0][0]

    return sentiment_label, confidence