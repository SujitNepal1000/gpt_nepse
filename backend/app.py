import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime
from db import engine
import io

# Load app metadata from environment if needed, otherwise use defaults
API_TITLE = os.getenv("API_TITLE", "NEPSE AI Trading System")
API_VERSION = os.getenv("API_VERSION", "1.0.0")

app = FastAPI(
    title=API_TITLE,
    description="Professional quantitative analytics and trading intelligence system for the Nepal Stock Exchange (NEPSE).",
    version=API_VERSION,
    contact={
        "name": "NEPSE AI Support",
        "url": "https://github.com/your-repo/nepsegpt",
    }
)

# CORS configuration
origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pipeline import process_with_history

@app.post("/upload_stock", tags=["Data Ingestion"], summary="Upload stock data")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel file containing stock data.
    The file must contain a 'Symbol' column.
    """
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

@app.get("/stocks", tags=["Dashboard"], summary="Get latest stock data")
def get_stocks():
    """
    Returns the latest stock data snapshot for all symbols.
    """
    query = """
    SELECT * FROM stock_data
    WHERE date = (SELECT MAX(date) FROM stock_data)
    """
    df = pd.read_sql(query, engine)
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/signals", tags=["Analysis"], summary="Get active trading signals")
def get_signals():
    """
    Returns stocks with active BUY or STRONG BUY signals.
    """
    df = pd.read_sql("SELECT * FROM stock_data", engine)
    df = df[df["signal"].isin(["BUY", "STRONG BUY"])]
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/sectors", tags=["Analysis"], summary="Get sector-wise distribution")
def get_sectors():
    """
    Returns the count of stocks per sector.
    """
    df = pd.read_sql("SELECT sector FROM stock_data", engine)
    return df.replace({np.nan: None}).groupby("sector").size().to_dict()