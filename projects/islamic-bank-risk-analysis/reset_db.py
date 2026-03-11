from backend.database import engine, Base
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def reset_db():
    try:
        print("Dropping all tables...")
        Base.metadata.drop_all(bind=engine)
        print("Recreating all tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Database reset successful.")
    except Exception as e:
        print(f"❌ Database Reset Error: {e}")

if __name__ == "__main__":
    reset_db()
