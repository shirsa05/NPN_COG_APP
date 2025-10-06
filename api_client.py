import requests
import streamlit as st

# IMPORTANT: Replace this placeholder with the actual URL from your Railway deployment.
API_URL = "https://npn-cognizant-hackathon.onrender.com/predict"
API_URL_2 = "https://hotel-review-analyzer.onrender.com/analyze"

def predict_sentiment_api(review_text: str):
    """
    Sends a review to the deployed model API and returns the prediction.

    Args:
        review_text: The string of the hotel review.

    Returns:
        A dictionary containing the 'label' and 'confidence', or None on error.
    """
    if not review_text or not review_text.strip():
        return None

    try:
        # The payload should match what your API endpoint expects.
        payload = {"text": review_text}
        
        # Make the POST request to your API
        response = requests.post(API_URL, json=payload, timeout=30) # 30-second timeout
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        # Parse the JSON response from the API
        api_response = response.json()
        
        # Extract data based on the new structure
        predicted_label = api_response.get("predicted_label")
        probabilities = api_response.get("probabilities")

        # Check if the expected keys are in the response
        if predicted_label is not None and probabilities and len(probabilities) == 2:
            if predicted_label == 1:
                label = "Happy"
                confidence = probabilities[1]  # The probability of the 'happy' class
            else:
                label = "Not Happy"
                confidence = probabilities[0]  # The probability of the 'not happy' class
            
            # Return the standardized dictionary that the Streamlit app expects
            return {"label": label, "confidence": confidence}
        else:
            st.error(f"API Error: The response from the model was not in the expected format. Received: {api_response}")
            return None

    except requests.exceptions.RequestException as e:
        # Display an informative error in the Streamlit app
        st.error(f"API Connection Error: Could not connect to the model endpoint. Please ensure the API is running. Details: {e}")
        return None

# # This function for aspect analysis remains the same
# def analyze_aspect_api(aspect_word: str):
#     """Calls the deployed API to analyze sentiment for a specific aspect."""

#     params = {"essential": aspect_word.lower().strip()}
#     try:
#         response = requests.get(API_URL_2, params=params, timeout=30)
#         response.raise_for_status()
#         return response.json()
        
#     except requests.exceptions.RequestException as e:
#         return {"error": f"API call failed: {e}"}