import streamlit as st
import pandas as pd
from datetime import datetime

# Import from your custom modules
from database import setup_database, insert_single_review, insert_bulk_reviews, fetch_all_reviews, get_aspect_counts
from dashboard import create_sentiment_distribution_plot, create_time_series_plot
from api_client import predict_sentiment_api

# --- 1. SETUP ---
st.set_page_config(page_title="Hotel Sentiment Analyzer", layout="wide")
setup_database()

# --- Initialize Session State ---
# This will store the loaded dataframe to prevent re-fetching on every rerun.
if 'all_reviews_df' not in st.session_state:
    st.session_state.all_reviews_df = None

# --- HELPER FUNCTION FOR DATE CLEANING (remains the same) ---
def normalize_timestamps(df, column_name='Time_Stamp'):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    failed_parses = df[column_name].isnull().sum()
    if failed_parses > 0:
        st.warning(f"{failed_parses} rows had a date format that could not be understood and were ignored.")
    df.dropna(subset=[column_name], inplace=True)
    return df

# --- 2. STREAMLIT UI ---
st.title("🏨 Hotel Review Sentiment Analyzer")
st.markdown("An intelligent dashboard to analyze hotel guest feedback, powered by a machine learning API.")

tab1, tab2, tab3, tab4 = st.tabs(["✍️ Single Review", "📤 Bulk Upload", "📈 Overall Dashboard", "🔬 Aspect Analysis"])

# --- TAB 1: SINGLE REVIEW ANALYSIS ---
with tab1:
    st.header("Analyze a Single Review")
    st.caption("Enter a single review to get an immediate sentiment prediction.")

    with st.form("single_review_form"):
        review_date = st.date_input("Date of Review")
        review_time = st.time_input("Time of Review")
        review_timestamp = datetime.combine(review_date, review_time)

        review_text = st.text_area("Enter the review text below:", height=150, placeholder="e.g., The room was wonderful and the staff were very helpful!")
        submitted = st.form_submit_button("Analyze Sentiment")

        if submitted and review_text.strip():
            with st.spinner("Contacting the model API..."):
                result = predict_sentiment_api(review_text)

            if result:
                prediction = result.get('label', 'Error')
                confidence = result.get('confidence', 0.0)
                if prediction == "Happy":
                    st.success(f"**Prediction: {prediction}**")
                else:
                    st.error(f"**Prediction: {prediction}**")

                label_for_db = 1 if prediction == "Happy" else 0
                insert_single_review(review_timestamp, review_text, label_for_db)
                st.info("✅ This review has been saved to the database.")
        elif submitted:
            st.warning("Please enter some review text.")

# --- TAB 2: BULK CSV UPLOAD ---
with tab2:
    st.header("Analyze a CSV File")
    st.caption("Upload a CSV file with 'Time_Stamp' and 'Description' columns for batch processing.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if 'Time_Stamp' in df_upload.columns and 'Description' in df_upload.columns:
                st.success("CSV file loaded successfully. Normalizing date formats...")
                df_upload = normalize_timestamps(df_upload, 'Time_Stamp')

                if st.button("Process and Save to Database"):
                    progress_bar = st.progress(0, text="Initializing analysis...")
                    total_rows = len(df_upload)
                    results = []

                    for index, row in df_upload.iterrows():
                        api_result = predict_sentiment_api(str(row['Description']))
                        results.append(api_result if api_result else {'label': 'Error', 'confidence': 0.0})
                        progress_value = (index + 1) / total_rows
                        progress_bar.progress(min(progress_value, 1.0), text=f"Analyzing review {index + 1}/{total_rows}")

                    progress_bar.empty()

                    df_results = pd.DataFrame(results)
                    df_upload['predicted_sentiment'] = df_results['label']
                    df_upload['confidence'] = df_results['confidence']
                    df_upload['predicted_label'] = df_upload['predicted_sentiment'].map({'Happy': 1, 'Not Happy': 0, 'Error': -1})

                    df_to_db = df_upload[df_upload['predicted_label'] != -1][['Time_Stamp', 'Description', 'predicted_label']].copy()
                    df_to_db.rename(columns={'Time_Stamp': 'timestamp', 'Description': 'review_text'}, inplace=True)
                    if not df_to_db.empty:
                        insert_bulk_reviews(df_to_db)

                    st.success("All reviews have been analyzed and saved to the database!")
                    st.divider()
                    st.header("Dashboard for Uploaded File")
                    st.plotly_chart(create_sentiment_distribution_plot(df_upload), use_container_width=True)

            else:
                st.error("Error: The CSV file must contain 'Time_Stamp' and 'Description' columns.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# --- TAB 3: OVERALL DASHBOARD ---
with tab3:
    st.header("Overall Sentiment Trends")
    st.caption("This dashboard shows the sentiment trend over time based on all reviews stored in the database.")

    if st.button("Load/Refresh Historical Data"):
        with st.spinner("Fetching and analyzing all historical reviews..."):
            st.session_state.all_reviews_df = fetch_all_reviews()
    
    st.divider()

    if st.session_state.all_reviews_df is not None and not st.session_state.all_reviews_df.empty:
        time_series_fig = create_time_series_plot(st.session_state.all_reviews_df)
        if time_series_fig:
            st.plotly_chart(time_series_fig, use_container_width=True)
        else:
            st.warning("Could not generate time-series plot. Ensure 'timestamp' column has valid dates.")
    else:
        st.info("Click 'Load/Refresh' or analyze some reviews!")


# --- TAB 4: ASPECT ANALYSIS ---
with tab4:
    st.header("Analyze Key Aspects from Your Database")
    st.caption("Enter a single word (e.g., 'staff', 'bed', 'location') to see its performance score based on all reviews in your database.")

    with st.form("aspect_analysis_form"):
        aspect_word = st.text_input("Enter aspect to analyze:", key="aspect_input").lower().strip()
        aspect_submitted = st.form_submit_button("Analyze Aspect")

        if aspect_submitted and aspect_word:
            with st.spinner(f"Searching database for reviews about '{aspect_word}'..."):
                counts = get_aspect_counts(aspect_word)

            if counts and counts["total_mentions"] > 0:
                total_mentions = counts["total_mentions"]
                happy_mentions = counts["happy_mentions"]
                not_happy_mentions = counts["not_happy_mentions"]
                performance_score = (happy_mentions / total_mentions) * 100

                st.subheader(f"Results for '{aspect_word}'")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Mentions", f"{total_mentions}")
                col2.metric("Happy Mentions", f"{happy_mentions}", "👍")
                col3.metric("Not Happy Mentions", f"{not_happy_mentions}", "👎")

                st.divider()
                st.subheader("Performance Score")
                st.progress(int(performance_score))

                if performance_score >= 75:
                    st.success(f"**Performance is Good with a score of {performance_score:.2f}%**")
                elif performance_score >= 50:
                    st.warning(f"**Performance is Neutral with a score of {performance_score:.2f}%**")
                else:
                    st.error(f"**Performance is Bad with a score of {performance_score:.2f}%. This may need improvement.**")
            
            elif counts and counts["total_mentions"] == 0:
                 st.warning(f"No reviews found in the database containing the word '{aspect_word}'.")
            else:
                st.error("Could not retrieve aspect analysis data at this time.")

        elif aspect_submitted:
            st.warning("Please enter an aspect to analyze.")
