# Vercel Deployment Guide - NEPSE AI System

Follow these steps to deploy your full-stack application to Vercel.

## 1. Prerequisites
- A Vercel account.
- A hosted PostgreSQL database (e.g., [Vercel Postgres](https://vercel.com/docs/storage/vercel-postgres), [Neon](https://neon.tech/), or [Supabase](https://supabase.com/)).

## 2. Environment Variables
You must set the following environment variables in your Vercel Project Settings:

| Variable | Value | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgresql://...` | Your Supabase Connection String (Transaction pooler recommended). |
| `API_TITLE` | `NEPSE AI Trading System` | Title for your Swagger UI. |
| `CORS_ORIGINS` | `https://your-vercel-domain.vercel.app` | Your frontend's production URL. |
| `REACT_APP_API_URL` | `/api` | (Optional) Defaults to `/api`. |

## 2.1 Supabase Configuration
1.  **Project Settings**: Go to your Supabase project -> Settings -> Database.
2.  **Connection String**: Choose the **Node.js** or **SQLAlchemy** compatible connection string.
3.  **Pooler**: If using Vercel, it is highly recommended to use the **Transaction Pooler** (port 6543) instead of the direct connection (port 5432).

## 3. Deployment Steps
1.  **Push to GitHub**: Push your local repository (containing `vercel.json` in the root) to GitHub.
2.  **Import to Vercel**:
    - Go to the Vercel Dashboard and click **New Project**.
    - Import your repository.
    - Vercel should automatically detect the `vercel.json` and configure the build.
3.  **Configure Build**:
    - Ensure the "Framework Preset" is set to "Other" or "Create React App" (Vercel usually handles this automatically based on `package.json`).
    - The `vercel.json` in the root will override default routing.

## 4. Troubleshooting
- **Serverless Size Limit**: If the deployment fails due to "Function too large", you may need to remove heavy dependencies like `xgboost` or use a smaller ML model.
- **Database Connection**: Ensure your database allows connections from Vercel's IP ranges or has "Allow all IPs" (shared database style) enabled if necessary.
