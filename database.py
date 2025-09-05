import streamlit as st
import psycopg2
import pandas as pd

# The @st.cache_resource decorator has been removed from this function
def get_db_connection():
    """Establishes a connection to the remote Postgres database."""
    try:
        # CORRECTED: Extract the single connection string from secrets
        connection_string = st.secrets["database"]["db_url"]
        # Pass the connection string as a single argument
        conn = psycopg2.connect(connection_string)
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"❌ Error connecting to the database: {e}")
        st.info("Please check your database credentials in Streamlit secrets and ensure the database is running.")
        return None

def setup_database():
    """Ensures the 'reviews' table exists in the database."""
    conn = get_db_connection()
    if conn:
        try:
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
        except Exception as e:
            st.error(f"Error during table setup: {e}")
        finally:
            conn.close()

def insert_single_review(timestamp, review, label):
    """Inserts a single review record into the database."""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as c:
                c.execute(
                    "INSERT INTO reviews (timestamp, review_text, predicted_label) VALUES (%s, %s, %s)",
                    (timestamp, review, label)
                )
            conn.commit()
        except Exception as e:
            st.error(f"Error inserting single review: {e}")
        finally:
            conn.close()

def insert_bulk_reviews(df):
    """Inserts a DataFrame of reviews into the database."""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as c:
                # Prepare data for efficient bulk insertion
                tuples = [tuple(x) for x in df.to_numpy()]
                cols = ','.join(list(df.columns))
                sql = f"INSERT INTO reviews ({cols}) VALUES %s"
                
                from psycopg2.extras import execute_values
                execute_values(c, sql, tuples)
            conn.commit()
        except Exception as e:
            st.error(f"Error during bulk insert: {e}")
        finally:
            conn.close()

def fetch_all_reviews():
    """Fetches all review records from the database."""
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql_query("SELECT timestamp, review_text, predicted_label FROM reviews ORDER BY timestamp DESC", conn)
            return df
        except Exception as e:
            st.error(f"Failed to fetch data from the database: {e}")
            return None
        finally:
            conn.close()