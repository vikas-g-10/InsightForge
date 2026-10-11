# InsightForge — AI-Powered Product Review Intelligence

InsightForge is an enterprise-grade product review intelligence platform that ingests unstructured customer feedback, classifies fine-grained sentiments and aspects, clusters issues into prioritized action items, simulates fix impacts, detects emerging anomalies, visually diagnoses hardware defects via Ollama vision models, deduces root-cause engineering hypotheses, and performs deep version and competitive product comparisons.

---

## Key Capabilities

1. **Review Ingestion & Privacy Normalization**
   - Resilient CSV & XLSX ingestion with automatic encoding detection and dialect sniffing.
   - PII redaction (`[email]`, `[phone]`) across all customer quotes and evidence fields.
   - Language detection and safety screening (Tier 1 physical hazards to Tier 3 mild warnings).
   - Rating-conflict detection, negation tracking, and sarcasm tagging.

2. **Aspect-Level Sentiment & Clustering**
   - Fine-grained clause splitting and aspect categorization (`battery`, `connectivity`, `build`, `audio`, `software`, etc.).
   - Issue clustering with traceable customer review citations.
   - Priority scoring (P0 Critical Safety to P3 Minor Annoyance) independent of overall volume.

3. **Fix-This-First Simulation**
   - Counterfactual simulation estimating star rating gains if targeted issues are resolved.
   - Team routing and triage assignments.

4. **Emerging-Issue Detection**
   - Compares recent vs. baseline time windows to identify velocity spikes and newly emerging defects.

5. **Defect Vision & Multimodal Image Diagnosis**
   - Integrates with local or private Ollama instances (`gemma3:4b`, `llava:7b`) for defect inspection.
   - Graceful degradation when Ollama is unavailable.

6. **Root-Cause Hypothesis Deduction**
   - Correlates review complaints and image observations with mechanical, electrical, firmware, and chemical failure modes.
   - Preserves strict disclaimer: engineering hypotheses only, requiring bench verification.

7. **Product and Version Comparison (NEW)**
   - **Version Comparison (`POST /api/compare/versions`)**:
     - Compare two releases of the same product (`Version A` vs `Version B`).
     - Distinguishes "not observed" from "resolved" when sample sizes are small ($<20$ reviews).
     - Tracks issue evolution: `improved`, `worsened`, `newly_observed`, `not_observed`, `resolved`, `unchanged`.
     - Analyzes aspect-level complaint deltas and safety escalations.
     - Evidence excerpts traceable to specific review IDs.
     - Evidence-grounded, non-causal observational summaries.
   - **Competitive Product Comparison (`POST /api/compare/products`)**:
     - Side-by-side comparison of two distinct products.
     - Evaluates sample balance (ratios $>4:1$ flagged) and small dataset sparsity.
     - Aspect strengths, weaknesses, and leader identification.
     - Shared complaint themes vs product-specific issues.
     - Safety alerts surfaced independently of overall volume.
     - Optional user-selected focus aspect (`battery`, `durability`, `software`, etc.).
     - Balanced trade-off summaries avoiding false "universal winner" declarations.

8. **Secure Supabase Authentication & User Data Isolation**
   - Official `@supabase/supabase-js` authentication in the React frontend.
   - Branded authentication view with real-time 3D React Three Fiber scene ("The Good. The Bad. The Insight.").
   - FastAPI server-side verification using Supabase JWKS (asymmetric RS256/ES256) with public key caching or symmetric HS256 secret.
   - All 14 application `/api/` endpoints are protected; only `GET /health` remains public for deployment probes.
   - Database isolation via Supabase Postgres table `public.analysis_history` with strict Row Level Security (RLS) policies enforcing `auth.uid() = user_id`.

9. **Machine Learning Training & Evaluation Pipeline (NEW)**
   - **Public Internet Dataset**: Amazon Reviews 2023 (`McAuley-Lab/Amazon-Reviews-2023`, Appliances subset) curated by UC San Diego (McAuley Lab).
   - **Proxy Sentiment Ground Truth**: Star ratings mapped to 3 sentiment classes: 1-2 stars $\rightarrow$ `negative`, 3 stars $\rightarrow$ `neutral`, 4-5 stars $\rightarrow$ `positive`.
   - **Algorithms Trained & Compared**:
     - *TF-IDF + Multinomial Logistic Regression* (`C=1.0`, `solver='lbfgs'`) — **Selected Champion**.
     - *TF-IDF + Calibrated Linear Support Vector Machine* (`LinearSVC` with probability calibration).
   - **Data Hygiene**: Stratified 70% Train (21,000), 15% Validation (4,500), 15% Test (4,500) split with fixed seed (`42`). Vectorizer fitted strictly on train set.
   - **Empirical Results (Held-Out Test Set)**:
     - Test Accuracy: **`70.82%`**
     - Macro F1: **`0.7080`** | Weighted F1: **`0.7080`**
     - Class F1: Negative (`0.7112`), Neutral (`0.6024`), Positive (`0.8104`)
     - Inference Latency: **`0.45 ms`** / review on CPU.
   - **Interactive Frontend Evaluation Page**:
     - Displays full model comparison, labeled $3 \times 3$ confusion matrix, class distributions, and live interactive prediction playground with calibrated probability distributions.

---

## API Endpoints & Access Control

| Method | Endpoint | Access | Description |
|---|---|---|---|
| `GET` | `/health` | **Public** | Service health status and runtime mode |
| `POST` | `/api/ingest/validate` | **Protected (Bearer)** | Validates review CSV/XLSX file columns and row counts |
| `POST` | `/api/analyze` | **Protected (Bearer)** | Executes core review analysis pipeline |
| `POST` | `/api/issues` | **Protected (Bearer)** | Groups reviews into clustered issues with priorities |
| `POST` | `/api/simulate` | **Protected (Bearer)** | Simulates star-rating impact of resolving selected issues |
| `POST` | `/api/emerging-issues` | **Protected (Bearer)** | Detects emerging spikes and new issue velocity |
| `GET` | `/api/vision/health` | **Protected (Bearer)** | Inspects Ollama connectivity and model status |
| `POST` | `/api/vision/analyze` | **Protected (Bearer)** | Multi-modal image defect diagnosis |
| `POST` | `/api/root-cause/hypothesize`| **Protected (Bearer)** | Generates traceable root-cause hypotheses |
| `POST` | `/api/compare/targets` | **Protected (Bearer)** | Discovers available products and versions in dataset |
| `POST` | `/api/compare/versions` | **Protected (Bearer)** | Compares two releases of the same product |
| `POST` | `/api/compare/products` | **Protected (Bearer)** | Compares two distinct products with trade-off analysis |
| `GET` | `/api/ml/metrics` | **Protected (Bearer)** | Retrieves held-out test evaluation metrics and confusion matrix |
| `POST` | `/api/ml/predict` | **Protected (Bearer)** | Interactive real-time ML sentiment and probability inference |

> **CORS Preflight**: `OPTIONS` requests across all routes bypass authentication to satisfy browser preflight checks.

> **CORS Preflight**: `OPTIONS` requests across all routes bypass authentication to satisfy browser preflight checks.

---

## Supabase Auth & Database Setup

### 1. Supabase Project Setup
1. Create a project at [supabase.com](https://supabase.com).
2. Under **Project Settings** -> **API**, locate your **Project URL** and **anon / public** key.
3. Under **Authentication** -> **URL Configuration**, set your **Site URL** and **Redirect URLs** (e.g. `http://localhost:5173/`, `https://yourdomain.com/`).

### 2. Apply Database Migration
Open the Supabase SQL Editor and execute [`supabase/migrations/20261011_create_analysis_history.sql`](file:///c:/Users/vikas/OneDrive/Documents/CBC%202.0/InsightForge/supabase/migrations/20261011_create_analysis_history.sql):
- Creates table `public.analysis_history` with foreign key referencing `auth.users(id) ON DELETE CASCADE`.
- Enables Row Level Security (RLS).
- Defines strict SELECT, INSERT, UPDATE, and DELETE policies enforcing `auth.uid() = user_id`.
- Revokes table access from the `anon` role.

---

## Running Locally

### Backend (FastAPI, Python 3.11+)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Configure backend environment variables in `.env` (or pass to shell):
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_JWT_AUDIENCE=authenticated
AUTH_ENABLED=true
```

### Frontend (React 19, TypeScript, Vite, Tailwind CSS v4)
```bash
cd frontend
npm ci
```

Create `frontend/.env.local`:
```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-publishable-key
```

Then start the development server:
```bash
npm run dev
```

### Machine Learning Pipeline & Optimization Experiment
```bash
cd backend

# 1. Acquire and preprocess Amazon Reviews 2023 dataset (Appliances category, 30,000 balanced reviews)
python -m ml.dataset --samples-per-class 10000

# 2. Train, optimize, and evaluate models on held-out test splits
python -m ml.train

# Generated versioned artifacts:
# - backend/ml/artifacts/sentiment_model.joblib
# - backend/ml/artifacts/evaluation_metrics.json
# - backend/ml/artifacts/EVALUATION_REPORT.md
```

#### Empirical Model Benchmark Results
| Experiment / Model | Val Macro F1 | Test Accuracy | Test Macro F1 | Status |
|---|---|---|---|---|
| **Optimized Calibrated Linear SVM (C=0.5, 20k TF-IDF, Stopwords Preserved)** | **0.7629** | **75.82%** | **0.7575** | **Selected Champion** |
| Soft Voting Ensemble (LR + Calibrated SVM, 20k TF-IDF) | 0.7615 | 75.89% | 0.7586 | Candidate |
| Optimized Logistic Regression (C=1.0, 20k TF-IDF) | 0.7585 | 75.60% | 0.7564 | Candidate |
| Original Baseline (TF-IDF 10k, English Stopwords, LR C=1.0) | 0.6997 | 70.82% | 0.7080 | Baseline |
| **Separate Binary Task (1-2 Stars Negative vs 4-5 Stars Positive, 3-Star Excluded)** | **0.9413** | **93.70%** | **0.9370** | **Binary Benchmark** |

### Running Test Suite
```bash
# Backend unit & integration tests (455 passed)
cd backend
python -m pytest

# Frontend lint & production build
cd ../frontend
npm run lint
npm run build
```

---

## License & Deployment
Refer to [`DEPLOYMENT.md`](file:///c:/Users/vikas/OneDrive/Documents/CBC%202.0/InsightForge/DEPLOYMENT.md) for full deployment instructions (Vercel for frontend, Render/Docker for backend).
