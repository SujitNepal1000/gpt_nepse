from fastapi import FastAPI
import pandas as pd
from db import engine

app = FastAPI()

@app.get("/stocks")
def get_stocks():
    query = """
    SELECT * FROM stock_data
    WHERE date = (SELECT MAX(date) FROM stock_data)
    """
    df = pd.read_sql(query, engine)
    return df.to_dict(orient="records")


@app.get("/signals")
def get_signals():
    df = pd.read_sql("SELECT * FROM stock_data", engine)
    df = df[df["signal"].isin(["BUY", "STRONG BUY"])]
    return df.to_dict(orient="records")


@app.get("/sectors")
def get_sectors():
    df = pd.read_sql("SELECT sector FROM stock_data", engine)
    return df.groupby("sector").size().to_dict()