from sqlalchemy import create_engine

DB_URL = "postgresql://postgres:admin@localhost:5432/nepsegpt"

engine = create_engine(DB_URL)