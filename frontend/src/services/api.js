import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000"
});

export const getStocks = () => API.get("/stocks");
export const getSignals = () => API.get("/signals");
export const getSectors = () => API.get("/sectors");