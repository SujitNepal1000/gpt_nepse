import axios from "axios";

const API = axios.create({
  // Use localhost in dev, or Vercel's relative /api route in production
  baseURL: "http://localhost:8000"
});

export const getStocks = () => API.get("/stocks");
export const getSignals = () => API.get("/signals");
export const getSectors = () => API.get("/sectors");