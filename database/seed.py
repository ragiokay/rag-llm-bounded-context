# BFR2: Data storage
# Downloads BPC dataset from HuggingFace and loads it into MySQL

import os
import mysql.connector
from datasets import load_dataset
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

def seed():
    print("Downloading BPC dataset from HuggingFace...")
    dataset = load_dataset("ibm-research/BPC")
    df = dataset["train"].to_pandas()
    print(f"Downloaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    conn = get_connection()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO bpc_dataset (phrase, question, answer, qid, situation, category, domain)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    rows = []
    for _, row in df.iterrows():
        rows.append((
            str(row.get("phrase", "")),
            str(row.get("question", "")),
            str(row.get("answer", "")),
            str(row.get("qid", "")),
            int(row.get("situation", 0)),
            str(row.get("category", "")),
            str(row.get("domain", ""))
        ))

    cursor.executemany(insert_sql, rows)
    conn.commit()

    print(f"Inserted {cursor.rowcount} rows into bpc_dataset")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    seed()