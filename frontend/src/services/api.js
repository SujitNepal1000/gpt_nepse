import axios from "axios";

const API = axios.create({
  // Use Vercel's relative /api route in production, or localhost in dev
  baseURL: process.env.REACT_APP_API_URL || "/api"
});

export const getStocks = () => API.get("/stocks");
export const getSignals = () => API.get("/signals");
export const getSectors = () => API.get("/sectors");