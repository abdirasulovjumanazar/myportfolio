import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

load_dotenv()

def create_database():
    # Connect to default 'postgres' database to create the new one
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='jfa37777',
        host='localhost',
        port='5432'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    try:
        cur.execute('CREATE DATABASE islamic_bank')
        print("✅ Database 'islamic_bank' created successfully.")
    except psycopg2.errors.DuplicateDatabase:
        print("ℹ️ Database 'islamic_bank' already exists.")
    except Exception as e:
        print(f"❌ Error creating database: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    create_database()
