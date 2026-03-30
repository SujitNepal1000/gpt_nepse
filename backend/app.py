from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime
from db import engine
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pipeline import process_with_history

@app.post("/upload_stock")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))
    
    # Standardize column names
    rename_map = {
        "Symbol": "symbol", "LTP": "ltp", "Close": "close", 
        "Vol": "volume", "Volume": "volume", "Open": "open", 
        "High": "high", "Low": "low", "Turnover": "turnover",
        "Diff %": "diff_pct", "Diff Pct": "diff_pct"
    }
    df.rename(columns=rename_map, inplace=True)
    
    if "symbol" not in df.columns:
        return {"error": "Excel file must contain a 'Symbol' column."}
    
    # Basic cleaning
    df["date"] = datetime.now().date()
    # Ensure numerical columns are clean
    for col in ["ltp", "close", "volume", "open", "high", "low", "turnover", "diff_pct"]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace(',', '').astype(float, errors='ignore')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Process with full technical indicators and ML signals
    df = process_with_history(df)
    
    if not df.empty:
        df.to_sql("stock_data", engine, if_exists="append", index=False)
        return {"message": f"Successfully uploaded and analyzed {len(df)} records!"}
    
    return {"error": "No data processed"}

@app.get("/stocks")
def get_stocks():
    query = """
    SELECT * FROM stock_data
    WHERE date = (SELECT MAX(date) FROM stock_data)
    """
    df = pd.read_sql(query, engine)
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/signals")
def get_signals():
    df = pd.read_sql("SELECT * FROM stock_data", engine)
    df = df[df["signal"].isin(["BUY", "STRONG BUY"])]
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/sectors")
def get_sectors():
    df = pd.read_sql("SELECT sector FROM stock_data", engine)
    return df.replace({np.nan: None}).groupby("sector").size().to_dict()