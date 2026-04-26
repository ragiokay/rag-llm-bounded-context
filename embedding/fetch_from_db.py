# IIR1: Database and Embedding Module interface
# Retrieves datasets from MySQL for embedding generation

import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

def fetch_all(limit=None):
    """
    Fetch all records from bpc_dataset.
    Returns a list of dicts.
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = "SELECT id, phrase, question, answer, category, domain FROM bpc_dataset"
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    print(f"[fetch_from_db] Fetched {len(rows)} rows from MySQL")
    return rows

def fetch_by_domain(domain):
    """
    Fetch records filtered by domain.
    Useful for domain-specific embedding collections.
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT id, phrase, question, answer, category, domain FROM bpc_dataset WHERE domain = %s",
        (domain,)
    )
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    print(f"[fetch_from_db] Fetched {len(rows)} rows for domain='{domain}'")
    return rows

def fetch_distinct_domains():
    """
    Returns list of all distinct domains in the dataset.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT domain FROM bpc_dataset ORDER BY domain")
    domains = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    return domains