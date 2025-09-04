import streamlit as st
import pandas as pd
from datetime import datetime
from predictor import load_assets, predict_sentiment
from database import setup_database, insert_single_review, insert_bulk_reviews, fetch_all_reviews
from dashboard import create_sentiment_distribution_plot, create_common_words_plot, create_time_series_plot

# --- 1. SETUP ---
st.set_page_config(page_title="Hotel Sentiment Analyzer", layout="wide")

# Load all assets: model, vectorizer, and NLTK components using Streamlit's cache
@st.cache_resource
def get_assets():
    """Loads all ML and NLTK assets once and caches them."""
    return load_assets()

model, vectorizer, lemmatizer, stop_words = get_assets()

# Initialize the database (creates the DB file and table if they don't exist)
setup_database()


# --- 2. STREAMLIT UI ---
st.title("🏨 Hotel Review Sentiment Analyzer")
st.markdown("Analyze hotel reviews to determine sentiment, powered by a machine learning model.")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Single Review Analysis", "Bulk CSV Upload", "Overall Dashboard"])

# --- Tab 1: Single Review ---
with tab1:
    st.header("Analyze a Single Review")
    
    # Use a form for better user experience
    with st.form("single_review_form"):
        review_date = st.date_input("Date of Review")
        review_time = st.time_input("Time of Review")
        review_text = st.text_area("Enter the review text below:", height=150, placeholder="The room was clean and the staff were very friendly...")
        submitted = st.form_submit_button("Analyze Sentiment")

        if submitted and review_text.strip() != "":
            # Combine date and time for the database timestamp
            review_timestamp = datetime.combine(review_date, review_time)
            
            # Call the prediction function from the predictor.py module
            label, confidence = predict_sentiment(review_text, vectorizer, model, lemmatizer, stop_words)
            
            # Display the result
            if label == 1:
                st.success(f"Prediction: Happy (Confidence: {confidence:.2%})")
            else:
                st.error(f"Prediction: Not Happy (Confidence: {confidence:.2%})")
            
            # Save the result to the database
            insert_single_review(review_timestamp, review_text, label)
            st.info("✅ Review has been saved to the database.")
        elif submitted:
            st.warning("Please enter some review text.")

# --- Tab 2: CSV Upload ---
with tab2:
    st.header("Analyze a CSV File")
    uploaded_file = st.file_uploader("Upload a CSV with 'Time_Stamp' and 'Description' columns", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully. Here's a preview:")
            st.dataframe(df_upload.head())
            
            if st.button("Process File and Generate Dashboard"):
                if 'Time_Stamp' in df_upload.columns and 'Description' in df_upload.columns:
                    with st.spinner("Analyzing reviews... This may take a moment."):
                        # Apply the prediction function to each row in the 'Description' column
                        results = df_upload['Description'].apply(
                            lambda text: predict_sentiment(str(text), vectorizer, model, lemmatizer, stop_words)
                        )
                        df_upload['predicted_label'] = [res[0] for res in results]
                        df_upload['confidence'] = [res[1] for res in results]
                        
                        # Prepare the DataFrame for database insertion
                        df_to_db = df_upload[['Time_Stamp', 'Description', 'predicted_label']].copy()
                        df_to_db.rename(columns={'Time_Stamp': 'timestamp', 'Description': 'review_text'}, inplace=True)
                        insert_bulk_reviews(df_to_db)

                    st.success("All reviews analyzed and saved to the database!")

                    # --- Display Dashboard using functions from dashboard.py ---
                    st.markdown("---")
                    st.header("Dashboard for Uploaded File")
                    
                    # Display sentiment distribution plot
                    fig_dist = create_sentiment_distribution_plot(df_upload)
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Display most common words plots in two columns
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_happy = create_common_words_plot(df_upload, 1, lemmatizer, stop_words)
                        st.plotly_chart(fig_happy, use_container_width=True)
                    with col2:
                        fig_not_happy = create_common_words_plot(df_upload, 0, lemmatizer, stop_words)
                        st.plotly_chart(fig_not_happy, use_container_width=True)

                else:
                    st.error("Error: CSV must contain 'Time_Stamp' and 'Description' columns.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- TAB 3: OVERALL DASHBOARD ---
with tab3:
    st.header("Overall Sentiment Trends")
    st.markdown("This dashboard shows the sentiment trend over time based on **all reviews** stored in the database.")
    
    if st.button("Load and Display Trends"):
        with st.spinner("Fetching and analyzing all historical reviews..."):
            all_reviews_df = fetch_all_reviews()
            
            if not all_reviews_df.empty:
                time_series_fig = create_time_series_plot(all_reviews_df)
                if time_series_fig:
                    st.plotly_chart(time_series_fig, use_container_width=True)
                else:
                    st.warning("Could not generate time-series plot. Not enough valid date entries found.")
            else:
                st.warning("No reviews found in the database yet. Analyze some reviews first!")

