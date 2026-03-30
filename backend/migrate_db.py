import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/nepsegpt")
engine = create_engine(DB_URL)

columns_to_add = [
    ("open", "FLOAT"),
    ("high", "FLOAT"),
    ("low", "FLOAT"),
    ("turnover", "FLOAT"),
    ("diff_pct", "FLOAT"),
    ("macd", "FLOAT"),
    ("macd_signal", "FLOAT"),
    ("macd_hist", "FLOAT"),
    ("ema20", "FLOAT"),
    ("ema50", "FLOAT"),
    ("bb_upper", "FLOAT"),
    ("bb_lower", "FLOAT"),
    ("atr", "FLOAT"),
    ("adx", "FLOAT"),
    ("di_plus", "FLOAT"),
    ("di_minus", "FLOAT"),
    ("mfi", "FLOAT"),
    ("stoch_k", "FLOAT"),
    ("williams_r", "FLOAT"),
    ("cci", "FLOAT"),
    ("smart_money", "VARCHAR"),
    ("trend_struct", "VARCHAR")
]

with engine.connect() as conn:
    for col_name, col_type in columns_to_add:
        try:
            conn.execute(text(f"ALTER TABLE stock_data ADD COLUMN {col_name} {col_type}"))
            print(f"Added column {col_name}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"Column {col_name} already exists")
            else:
                print(f"Error adding {col_name}: {e}")
    conn.commit()
print("Migration complete")
