from app import app, db
from sqlalchemy import text


def fix_chromadb_schema():
    with app.app_context():
        try:
            db.session.execute(text("ALTER TABLE collections ADD COLUMN topic TEXT;"))
            db.session.commit()
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                print(f"Error: {e}")


if __name__ == "__main__":
    fix_chromadb_schema()
