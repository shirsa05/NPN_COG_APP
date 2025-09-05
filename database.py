import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# --- 1. DATABASE CONNECTION SETUP ---
@st.cache_resource
def get_db_connection():
    """Establishes a connection to the Neon Postgres database using a connection string."""
    try:
        # Connect using the single URL from secrets
        conn = psycopg2.connect(st.secrets.database.db_url)
        return conn
    except Exception as e:
        # We log the error in the terminal, but show a user-friendly message in the app.
        print(f"Database connection error: {e}")
        st.error("Could not connect to the database. Please check the credentials in your secrets file.")
        return None

# --- 2. CORE DATABASE FUNCTIONS ---

def setup_database():
    """
    Connects to the database and ensures the 'reviews' table exists.
    This is safe to run on every app startup.
    """
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as c:
                c.execute('''
                    CREATE TABLE IF NOT EXISTS reviews (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP,
                        review_text TEXT,
                        predicted_label INTEGER
                    )
                ''')
            conn.commit()
            conn.close()
    except Exception:
        # Pass silently on startup if secrets aren't available yet or connection fails.
        # The error will be shown to the user when a DB action is attempted.
        pass

def insert_single_review(timestamp, review, label):
    """Inserts a single review record into the Neon database."""
    conn = get_db_connection()
    if conn:
        with conn.cursor() as c:
            sql = "INSERT INTO reviews (timestamp, review_text, predicted_label) VALUES (%s, %s, %s)"
            c.execute(sql, (timestamp, review, label))
        conn.commit()
        conn.close()

def insert_bulk_reviews(df):
    """Inserts a DataFrame of reviews into the Neon database using an efficient bulk method."""
    conn = get_db_connection()
    if conn and not df.empty:
        with conn.cursor() as c:
            # Prepare data for efficient bulk insertion
            tuples = [tuple(x) for x in df.to_numpy()]
            cols = ','.join(list(df.columns))
            sql = f"INSERT INTO reviews ({cols}) VALUES %s"
            
            # Use execute_values for a fast bulk insert
            execute_values(c, sql, tuples)
        conn.commit()
        conn.close()

def fetch_all_reviews():
    """Fetches all review records from the Neon database and returns them as a DataFrame."""
    conn = get_db_connection()
    if conn:
        try:
            # Read all data from the reviews table, ordering by the newest first
            df = pd.read_sql("SELECT * FROM reviews ORDER BY timestamp DESC", conn)
            return df
        except Exception as e:
            st.error(f"Failed to fetch data from the database: {e}")
        finally:
            conn.close()
    # Return an empty DataFrame if the connection fails
    return pd.DataFrame()
