import streamlit as st
import pandas as pd
from datetime import datetime

# Import from your custom modules
from database import setup_database, insert_single_review, insert_bulk_reviews, fetch_all_reviews
from dashboard import create_sentiment_distribution_plot, create_time_series_plot
from api_client import predict_sentiment_api, analyze_aspect_api

# --- 1. SETUP ---
st.set_page_config(page_title="Hotel Sentiment Analyzer", layout="wide")

# Initialize the database and its table
setup_database()

# --- NEW: HELPER FUNCTION FOR DATE STANDARDIZATION ---
def normalize_timestamps(df, column_name='Time_Stamp'):
    """
    Intelligently converts a column with mixed date formats into a
    standardized datetime format that the database and plots can use.
    """
    # pd.to_datetime is excellent at guessing formats like dd-mm-yyyy, mm/dd/yyyy, etc.
    # `errors='coerce'` will replace any unparseable dates with NaT (Not a Time).
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce', dayfirst=False)
    
    # Count how many dates could not be parsed
    failed_parses = df[column_name].isnull().sum()
    if failed_parses > 0:
        st.warning(f"{failed_parses} rows had a date format that could not be understood. These rows will be ignored.")
    
    # Remove rows where the date could not be understood to ensure data quality
    df.dropna(subset=[column_name], inplace=True)
    return df

# --- 2. STREAMLIT UI ---
st.title("🏨 Hotel Review Sentiment Analyzer")
st.markdown("This application analyzes hotel reviews to determine sentiment by calling a deployed machine learning API.")

# Create the main tabs for the application
tab1, tab2, tab3, tab4 = st.tabs(["Single Review Analysis", "Bulk CSV Upload", "Overall Dashboard", "Aspect Analysis"])


# --- TAB 1: SINGLE REVIEW ANALYSIS ---
with tab1:
    st.header("Analyze a Single Review")
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
                prediction = result.get('label', 'Error') # Use .get for safety
                confidence = result.get('confidence', 0.0)

                if prediction == "Happy":
                    st.success(f"Prediction: {prediction}")
                else:
                    st.error(f"Prediction: {prediction}")

                label_for_db = 1 if prediction == "Happy" else 0
                insert_single_review(review_timestamp, review_text, label_for_db)
                st.info("✅ This review has been saved to the database.")
        elif submitted:
            st.warning("Please enter some review text.")


# --- TAB 2: BULK CSV UPLOAD (UPDATED) ---
with tab2:
    st.header("Analyze a CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with 'Time_Stamp' and 'Description' columns", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if 'Time_Stamp' in df_upload.columns and 'Description' in df_upload.columns:
                st.success("CSV file loaded successfully. Here's a preview:")
                st.dataframe(df_upload.head())

                # --- APPLY THE NEW DATE NORMALIZATION STEP ---
                with st.spinner("Standardizing date formats in 'Time_Stamp' column..."):
                    df_upload = normalize_timestamps(df_upload, 'Time_Stamp')
                # -------------------------------------------

                if st.button("Process and Save to Database"):
                    progress_bar = st.progress(0, text="Initializing analysis...")
                    total_rows = len(df_upload)
                    results = []

                    for index, row in df_upload.iterrows():
                        api_result = predict_sentiment_api(str(row['Description']))
                        results.append(api_result if api_result else {'label': 'Error', 'confidence': 0.0})
                        progress_value = (index + 1) / total_rows
                        # Ensure the value never exceeds 1.0 before updating the bar
                        progress_bar.progress(min(progress_value, 1.0), text=f"Analyzing review {index + 1}/{total_rows}")
                        # ---------------
                    
                    progress_bar.empty()
                    
                    df_results = pd.DataFrame(results)
                    df_upload['predicted_sentiment'] = df_results['label']
                    df_upload['confidence'] = df_results['confidence']
                    df_upload['predicted_label'] = df_results['label']
                    
                    df_upload['predicted_label'] = df_upload['predicted_sentiment'].map({'Happy': 1, 'Not Happy': 0, 'Error': -1})

                    df_to_db = df_upload[df_upload['predicted_label'] != -1][['Time_Stamp', 'Description', 'predicted_label']].copy()
                    df_to_db.rename(columns={'Time_Stamp': 'timestamp', 'Description': 'review_text'}, inplace=True)
                    if not df_to_db.empty:
                        insert_bulk_reviews(df_to_db)
                    
                    st.success("All reviews have been analyzed and saved to the database!")
                    
                    st.markdown("---")
                    st.header("Dashboard for Uploaded File")
                    st.plotly_chart(create_sentiment_distribution_plot(df_upload), use_container_width=True)
            else:
                st.error("Error: The CSV file must contain 'Time_Stamp' and 'Description' columns.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")


# --- TAB 3: OVERALL DASHBOARD ---
with tab3:
    st.header("Overall Sentiment Trends")
    st.markdown("This dashboard shows the sentiment trend over time based on **all reviews** stored in the database.")
    
    if st.button("Load and Display Trends"):
        with st.spinner("Fetching and analyzing all historical reviews..."):
            all_reviews_df = fetch_all_reviews()
            
            if all_reviews_df is not None and not all_reviews_df.empty:
                time_series_fig = create_time_series_plot(all_reviews_df)
                if time_series_fig:
                    st.plotly_chart(time_series_fig, use_container_width=True)
                else:
                    st.warning("Could not generate time-series plot. Ensure 'timestamp' column has valid dates.")
            else:
                st.warning("No reviews found in the database yet. Analyze some reviews first!")

# --- NEW TAB 4: ASPECT ANALYSIS ---
with tab4:
    st.header("Analyze Key Aspects")
    st.info("Enter a single word (e.g., 'staff', 'bed', 'location') to see its performance score based on the model's knowledge.")

    aspect_word = st.text_input("Enter aspect to analyze:", key="aspect_input")

    if st.button("Analyze Aspect", key="aspect_button"):
        if aspect_word:
            with st.spinner(f"Analyzing aspect: '{aspect_word}'..."):
                aspect_result = analyze_aspect_api(aspect_word)
                
                if "error" in aspect_result:
                    st.error(aspect_result["error"])
                elif "performance_score" in aspect_result:
                    performance = aspect_result["performance_score"][0]
                    score = aspect_result["performance_score"][1]
                    
                    if performance == "Good":
                        st.success(f"Performance for '{aspect_word}' is **Good**, Performance score of '{aspect_word}': {score:.2f}%.")
                    else:
                        st.error(f"Performance for '{aspect_word}' is **Bad**. This may need improvement. Performance score of '{aspect_word}': {score:.2f}%.")
        else:
            st.warning("Please enter an aspect to analyze.")

