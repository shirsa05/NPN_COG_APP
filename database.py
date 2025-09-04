import sqlite3
import pandas as pd

DB_FILE = "reviews.db"

def setup_database():
    """Creates the reviews table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            review_text TEXT,
            predicted_label INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def insert_single_review(timestamp, review, label):
    """Inserts a single review record into the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO reviews (timestamp, review_text, predicted_label) VALUES (?, ?, ?)",
              (timestamp, review, label))
    conn.commit()
    conn.close()

def insert_bulk_reviews(df):
    """Inserts a DataFrame of reviews into the database."""
    conn = sqlite3.connect(DB_FILE)
    df.to_sql('reviews', conn, if_exists='append', index=False)
    conn.close()
    
def fetch_all_reviews():
    """Fetches all review records from the database."""
    conn = sqlite3.connect(DB_FILE)
    try:
        # Select only the columns needed for the time-series plot
        df = pd.read_sql_query("SELECT timestamp, predicted_label FROM reviews", conn)
        return df
    except Exception as e:
        # Return an empty DataFrame if the table doesn't exist or an error occurs
        print(f"Database read error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()