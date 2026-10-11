# InsightForge Deployment Guide

This guide details how to deploy InsightForge to production:
- **Frontend**: Hosted on [Vercel](https://vercel.com) (optimized for static Vite SPA assets and global CDN distribution).
- **Backend**: Hosted on a persistent Python runtime like [Render](https://render.com), [Railway](https://railway.app), [Fly.io](https://fly.io), or Docker container service.

> **Architecture Note**: Do not deploy the existing FastAPI backend directly onto Vercel Serverless Functions. InsightForge runs CPU-bound NLP pipelines (language detection, rule-based tokenizers, clause analysis, clustering algorithms, and simulation modeling) with persistent threadpools and file uploads that require a dedicated Python ASGI server process (`uvicorn`).

---

## 1. Backend Deployment (Python Hosting)

### Option A: Render (Web Service)
1. Fork or push the InsightForge repository to GitHub.
2. In Render, select **New Web Service** and connect your repository.
3. Configure the service settings:
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Set Environment Variables in Render:
   - `APP_MODE`: `lite`
   - `ALLOWED_ORIGINS`: Your Vercel frontend URL (e.g. `https://insightforge.vercel.app`)
   - `SUPABASE_URL`: Your Supabase Project URL (e.g. `https://xyzcompany.supabase.co`)
   - `SUPABASE_JWT_AUDIENCE`: `authenticated`
   - `AUTH_ENABLED`: `true`
   - `SUPABASE_JWT_SECRET`: *(Optional)* Only if using legacy HS256 secret. NEVER expose to frontend.
5. Deploy and copy your backend URL (e.g. `https://insightforge-api.onrender.com`).
6. Verify deployment by visiting `https://insightforge-api.onrender.com/health`. You should see `{"status":"ok","mode":"lite","db_connected":false}`.

### Option B: Docker / Container Platform (Railway, Fly.io, AWS App Runner)
A production-ready [`Dockerfile`](file:///c:/Users/vikas/OneDrive/Documents/CBC%202.0/InsightForge/backend/Dockerfile) is provided in `backend/`:
```bash
docker build -t insightforge-api ./backend
docker run -p 8000:8000 \
  -e APP_MODE=lite \
  -e ALLOWED_ORIGINS="https://insightforge.vercel.app" \
  -e SUPABASE_URL="https://your-project.supabase.co" \
  -e SUPABASE_JWT_AUDIENCE="authenticated" \
  -e AUTH_ENABLED="true" \
  insightforge-api
```

---

## 2. Frontend Deployment (Vercel)

### Deploying via Vercel Dashboard
1. In Vercel, click **Add New...** -> **Project** and select your repository.
2. In **Project Settings**:
   - **Framework Preset**: `Vite`
   - **Root Directory**: Select `frontend` (or leave as root if using the root `vercel.json`)
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
3. Add Environment Variables:
   - `VITE_API_BASE_URL`: The URL of your deployed backend without trailing slash (e.g. `https://insightforge-api.onrender.com`).
   - `VITE_SUPABASE_URL`: Your Supabase Project URL (e.g. `https://xyzcompany.supabase.co`).
   - `VITE_SUPABASE_ANON_KEY`: Your project public anon key (safe for browser client).
4. Click **Deploy**.
5. Once deployed, copy your production Vercel domain (e.g. `https://insightforge.vercel.app`).
6. **Ensure CORS Alignment**: If your backend `ALLOWED_ORIGINS` was configured with a placeholder, update `ALLOWED_ORIGINS` in your backend environment variables to match your exact Vercel URL.
7. **Configure Supabase Redirects**: In Supabase Dashboard -> **Authentication** -> **URL Configuration**, add your Vercel URL (e.g. `https://insightforge.vercel.app/`) to the **Redirect URLs** list so password reset and email confirmation tokens resolve properly.

---

## 3. Database Migration & Row Level Security (RLS)

1. In Supabase Dashboard, go to **SQL Editor**.
2. Run the migration script located at [`supabase/migrations/20261011_create_analysis_history.sql`](file:///c:/Users/vikas/OneDrive/Documents/CBC%202.0/InsightForge/supabase/migrations/20261011_create_analysis_history.sql).
3. The migration sets up:
   - Table `public.analysis_history` with foreign key cascade to `auth.users(id)`.
   - Indexes on `(user_id, created_at desc)`.
   - RLS enabled with 4 strict policies enforcing `auth.uid() = user_id` for SELECT, INSERT, UPDATE, and DELETE.
   - Revokes permissions from the anonymous role.

---

## 3. Advanced Intelligence & Vision Inference Setup

### Local Development Setup (Ollama Vision)
1. **Install and Start Ollama**:
   Install Ollama from [ollama.com](https://ollama.com) and start the local daemon:
   ```bash
   ollama serve
   ```
2. **Pull a Compatible Vision Model**:
   InsightForge defaults to `gemma3:4b`, which natively supports multimodal vision input:
   ```bash
   ollama pull gemma3:4b
   ```
   *(Alternative tested vision models: `llava:7b`, `llama3.2-vision:11b`).*
3. **Verify Connectivity**:
   Visit `http://localhost:8000/api/vision/health` or check the **Defect Vision** page in the UI. When Ollama is active with a vision-capable model, the system displays a green `healthy` indicator.

### Production Deployment Architecture Notice (Crucial)
> ⚠️ **Localhost Isolation in Production**:
> Ollama running on a developer's local machine (`http://localhost:11434`) is **not accessible** to a cloud-hosted backend container (e.g., Render, Railway, Fly.io, or AWS).
>
> If you deploy InsightForge to the cloud without configuring an external vision service:
> - **Graceful Degradation / Fully Optional**: The core review analysis pipeline, aspect sentiment classification, issue clustering, prioritization, Fix-This-First simulator, emerging-issue detection, and review-based root cause hypothesis generator remain **100% operational**.
> - **Honest Status Reporting**: The backend **never silently pretends** image inference works. `GET /api/vision/health` will return `status: "unavailable"` and the UI will clearly guide the user that image inference requires an accessible Ollama endpoint.
>
> **Connecting Vision in Production**:
> To enable multimodal image defect analysis in production:
> 1. Host Ollama on a GPU-enabled cloud container or VM (e.g., AWS EC2 G-instance, RunPod, Modal, or private Kubernetes cluster).
> 2. Ensure Ollama is protected behind private networking (VPC peering, Tailscale, or an API gateway with mTLS/reverse proxy). **Never expose Ollama publicly without access controls.**
> 3. Set the following backend environment variables:
>    ```env
>    OLLAMA_BASE_URL="https://private-ollama.internal.yourdomain.com"
>    OLLAMA_MODEL="gemma3:4b"
>    OLLAMA_TIMEOUT_SECONDS=45.0
>    MAX_IMAGE_UPLOAD_BYTES=10485760
>    ```

---

## 4. Local Development

### Terminal 1: Backend
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

### Terminal 2: Frontend
```bash
cd frontend
npm run dev
```
In development, the Vite dev server proxies requests from `/api/*` and `/health` to `http://127.0.0.1:8000`. You can also test cross-origin requests by setting `VITE_API_BASE_URL=http://localhost:8000` in `frontend/.env.local`.

---

## 5. Product & Version Comparison Endpoints

InsightForge comparison capabilities are exposed via dedicated, stateless endpoints:
- `POST /api/compare/targets`: Scans dataset for available products and versions.
- `POST /api/compare/versions`: Evaluates version evolution with sample size sensitivity (distinguishing "not observed" from "resolved").
- `POST /api/compare/products`: Conducts competitive product trade-off benchmarking without universal winner bias.

Both endpoints operate in-memory on ingested or pre-analyzed batches, adhering strictly to the `APP_MODE=lite` architecture without requiring database persistence or external model dependencies.

