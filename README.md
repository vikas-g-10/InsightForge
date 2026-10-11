# InsightForge — AI-Powered Product Review Intelligence

InsightForge is an AI-powered product review analysis platform that helps businesses understand what customers are saying about their products, discover recurring issues, identify potential safety concerns, and make better product-improvement decisions.

Instead of manually reading thousands of customer reviews, InsightForge transforms customer feedback into structured insights, prioritized issues, sentiment analysis, and evidence-backed recommendations.

It combines a rule-based natural language processing pipeline, a trained machine-learning sentiment classifier, and a local Ollama vision model to analyze both written feedback and product images.

## Table of Contents

- [The Problem](#the-problem)
- [What InsightForge Does](#what-insightforge-does)
- [Key Features](#key-features)
- [Machine Learning Training](#machine-learning-training)
- [Local AI with Ollama Gemma 3 4B](#local-ai-with-ollama-gemma-3-4b)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [Training the Sentiment Model](#training-the-sentiment-model)
- [API Overview](#api-overview)
- [Deployment](#deployment)
- [Limitations](#limitations)

---

## The Problem

Product teams receive large amounts of customer feedback through online reviews, ratings, and product images. Manually analyzing this information is time-consuming and makes it difficult to identify the most important problems.

Common challenges include:

- Understanding customer sentiment beyond an overall star rating.
- Identifying specific product aspects that customers complain about.
- Finding recurring defects across large numbers of reviews.
- Recognizing potentially serious safety concerns that should not be overlooked.
- Determining whether problems are becoming more frequent.
- Understanding whether a newer product version has improved or introduced new issues.
- Investigating possible causes of product failures.
- Turning unstructured feedback into actionable engineering priorities.

**InsightForge addresses these challenges by converting unstructured product feedback into organized, traceable product intelligence.**

## What InsightForge Does

The platform follows a practical analysis workflow:

1. **Upload:** Import customer reviews from CSV or Excel files.
2. **Normalize:** Clean review text and redact common email addresses and phone numbers.
3. **Analyze:** Identify sentiment, product aspects, language signals, and potential safety concerns.
4. **Discover:** Group related complaints into issues and prioritize them.
5. **Investigate:** Examine emerging problems, compare products or versions, and generate possible root-cause hypotheses.
6. **Visualize:** Explore the findings through the web interface.
7. **Evaluate:** Test the separately trained sentiment classifier and inspect its evaluation metrics.

## Key Features

### 1. Review Ingestion and Privacy Normalization

- CSV and XLSX file uploads.
- Automatic encoding and delimiter detection for supported CSV formats.
- Review-text column detection with support for selecting a column when necessary.
- Removal of empty or unusable review entries.
- Best-effort redaction of common email addresses and phone numbers.
- Detection of rating conflicts, negation, sarcasm-related signals, and language indicators.

Privacy normalization is a protective measure, not a guarantee that every personally identifiable detail will be removed.

### 2. Aspect-Based Sentiment Analysis

Analyze feedback about individual product aspects rather than relying only on an overall rating.

Examples of aspects include:

- Battery and power
- Connectivity
- Build quality and durability
- Audio
- Software and usability

The NLP pipeline identifies relevant clauses and aspect-related evidence, helping distinguish what customers like from what they dislike.

### 3. Issue Clustering and Prioritization

Group related complaints into identifiable issues and provide evidence from the underlying reviews.

Issues receive priority levels ranging from:

- **P0:** Critical safety concern
- **P1:** High-priority issue
- **P2:** Moderate issue
- **P3:** Minor annoyance

Prioritization is designed to consider issue severity, not simply the number of reviews mentioning it.

Safety-related signals should be reviewed and verified by appropriate people rather than treated as confirmed findings.

### 4. Fix-This-First Simulation

Explore the potential impact of addressing selected issues.

- Estimate possible changes in customer star ratings.
- Rank issues to help teams decide what to investigate first.
- Support triage and team-routing decisions.

These outputs are estimates based on the available data, not guarantees of future customer behavior.

### 5. Emerging-Issue Detection

Compare recent and baseline review periods to identify:

- Newly observed issues.
- Increasing complaint frequency.
- Accelerating issue patterns.
- Safety-related escalations.

This helps teams investigate potential product problems before they become widespread.

### 6. Product Defect Vision with Ollama

Analyze uploaded product images using a locally hosted, vision-capable Ollama model.

- Inspect images for visible defects.
- Combine image observations with review complaints during root-cause investigation.
- Check whether the configured Ollama model is available.
- Continue supporting the core review-analysis features when Ollama is unavailable.

### 7. Root-Cause Hypothesis Generation

Combine review evidence and available image observations to generate possible explanations for product failures.

Potential areas of investigation include mechanical, electrical, firmware, and material-related failure modes.

**Important:** These are engineering hypotheses, not confirmed diagnoses. Physical inspection and appropriate testing are necessary to verify them.

### 8. Product Version Comparison

Compare two versions of the same product to understand how reported issues change.

The analysis can identify issues that are:

- Improved
- Worsened
- Newly observed
- Not observed
- Resolved
- Unchanged

It also considers sample-size limitations and provides review evidence to support the comparison.

A problem missing from a small set of reviews is not automatically considered resolved.

### 9. Competitive Product Comparison

Compare two products side by side.

- Compare aspect-level strengths and weaknesses.
- Identify shared complaint themes and product-specific issues.
- Examine differences in available review samples.
- Highlight safety concerns independently of overall complaint volume.
- Provide trade-off summaries instead of automatically declaring one product superior in every respect.

### 10. Trained Sentiment Model and Evaluation Dashboard

InsightForge includes a separately trained machine-learning model for classifying written reviews as negative, neutral, or positive.

The evaluation interface provides:

- Model comparison results.
- A confusion matrix.
- Per-class evaluation metrics.
- Sentiment class distributions.
- An interactive prediction interface with class-probability estimates.

### 11. Authentication and User Data Isolation

The application integrates Supabase authentication.

- Frontend authentication using Supabase.
- Server-side verification of access tokens.
- Protected application API endpoints.
- A Supabase Postgres analysis-history table with Row Level Security (RLS) policies.

Follow the configuration steps below before running the authenticated application.

---

## Machine Learning Training

InsightForge contains a supervised machine-learning pipeline developed specifically for **text-based sentiment classification**.

### Dataset

The model-training pipeline uses a curated subset of the Appliances category from Amazon Reviews 2023.

- **Dataset:** Amazon Reviews 2023
- **Source:** [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Curated sample:** 30,000 reviews
- **Purpose:** Train and evaluate a three-class sentiment classifier

The star ratings are used as proxy labels:

| Star rating | Assigned sentiment |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

These are proxy labels because a customer's star rating does not always match the sentiment expressed in the written review.

### Training and evaluation procedure

The training pipeline performs the following steps:

1. Acquires and preprocesses a balanced dataset.
2. Converts review text into numerical TF-IDF features.
3. Creates stratified training, validation, and test splits.
4. Trains and compares multiple candidate classifiers.
5. Selects the champion model using validation macro F1.
6. Evaluates the selected model on the held-out test set.
7. Saves the trained pipeline and evaluation results.

The three-way split is:

| Split | Samples | Percentage |
|---|---:|---:|
| Training | 21,000 | 70% |
| Validation | 4,500 | 15% |
| Test | 4,500 | 15% |
| **Total** | **30,000** | **100%** |

A fixed random seed of `42` is used for reproducibility. The TF-IDF vectorizer is fitted on the training split rather than on the complete dataset.

### Models compared

The training experiment compares:

- TF-IDF with Logistic Regression as a baseline.
- An optimized TF-IDF representation with Logistic Regression.
- An optimized TF-IDF representation with a probability-calibrated Linear Support Vector Machine (SVM).
- A soft-voting ensemble combining Logistic Regression and the calibrated SVM.

The selected champion is:

**Optimized TF-IDF + Calibrated Linear SVM (`C=0.5`)**

Its TF-IDF representation uses up to 20,000 features, including unigrams and bigrams, while preserving stopwords that can carry important sentiment information.

### Evaluation results

Results from the saved evaluation report:

| Metric | Three-class model |
|---|---:|
| Held-out test accuracy | 75.82% |
| Macro F1 score | 0.7575 |
| Weighted F1 score | 0.7575 |
| Average inference latency | Approximately 1.93 ms per review |

Per-class F1 scores:

| Sentiment | F1 score |
|---|---:|
| Negative | 0.7531 |
| Neutral | 0.6610 |
| Positive | 0.8585 |

The neutral class is more difficult to classify because three-star reviews often contain mixed opinions. For example, a customer may praise a product's performance while complaining about its durability.

A separate binary experiment excludes three-star reviews and distinguishes negative from positive reviews. It achieved **93.70% test accuracy** on its separate 20,000-review dataset. This result should not be compared directly with the three-class result because the task and dataset are different.

### Saved training artifacts

The training pipeline creates the following files:

- `backend/ml/artifacts/sentiment_model.joblib` — serialized trained model pipeline.
- `backend/ml/artifacts/evaluation_metrics.json` — model metrics and comparison results.
- `backend/ml/artifacts/EVALUATION_REPORT.md` — detailed evaluation report.

The trained sentiment classifier is a conventional supervised machine-learning model. It is **not a fine-tuned Gemma or other large language model**.

---

## Local AI with Ollama Gemma 3 4B

InsightForge supports local multimodal inference through [Ollama](https://ollama.com).

The configured default vision model is **`gemma3:4b`**, a vision-capable Gemma model that can process text and images.

### What Gemma does in InsightForge

Gemma 3 4B is used for the product-image defect analysis feature. It can help examine visible product conditions and produce observations that may be considered alongside customer complaints.

The integration includes:

- Local Ollama connectivity checks.
- Configurable model and endpoint settings.
- Image-analysis requests to the configured vision model.
- Clear health reporting when Ollama is unavailable.

### Install and configure Ollama

Install Ollama using the instructions on [the official Ollama website](https://ollama.com).

Start the Ollama service if it is not already running:

```bash
ollama serve
```

In another terminal, download the model:

```bash
ollama pull gemma3:4b
```

Check that the model is installed:

```bash
ollama list
```

Ollama normally exposes its local API at:

```text
http://localhost:11434
```

InsightForge's root `.env` example uses the following configuration:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_TIMEOUT_SECONDS=30.0
MAX_IMAGE_UPLOAD_BYTES=10485760
```

Restart the backend after changing its environment configuration.

### How the AI components differ

InsightForge uses two distinct types of AI models:

| Component | Purpose | How it is obtained |
|---|---|---|
| TF-IDF + Calibrated Linear SVM | Predicts sentiment from written review text | Trained and evaluated for this project |
| Ollama Gemma 3 4B | Analyzes product images for visible defect evidence | Downloaded and run locally through Ollama |

Gemma 3 4B is an optional local vision component. The core review-analysis pipeline does not require it to be available.

**Local versus cloud deployment:** A cloud-hosted backend cannot access Ollama at `localhost:11434` on your personal computer. To use vision analysis in a deployed environment, configure the backend to reach a properly secured, accessible Ollama instance. Do not expose an Ollama service publicly without appropriate access controls.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React 19, TypeScript, Vite |
| UI and styling | Tailwind CSS v4, Lucide React, Motion |
| Data visualization | Recharts |
| 3D authentication experience | Three.js, React Three Fiber |
| Backend API | Python, FastAPI, Uvicorn |
| Data processing | Pandas, OpenPyXL |
| NLP pipeline | Rule-based text normalization, clause splitting, aspect detection, sentiment lexicon, safety scanning |
| Machine learning | Scikit-learn, TF-IDF, Logistic Regression, calibrated Linear SVM |
| Model persistence | Joblib |
| Local vision inference | Ollama, Gemma 3 4B |
| Authentication | Supabase Auth |
| Database | Supabase Postgres with Row Level Security |
| Deployment | Vercel frontend, Render or another suitable Python hosting service |

---

## Project Structure

```text
InsightForge/
├── backend/
│   ├── main.py                    # FastAPI application and API endpoints
│   ├── auth.py                    # Server-side authentication
│   ├── ingest.py                  # CSV/XLSX ingestion and validation
│   ├── ml/
│   │   ├── dataset.py             # Dataset acquisition and preprocessing
│   │   ├── train.py               # Training and evaluation pipeline
│   │   ├── model_service.py       # Loads model and predicts sentiment
│   │   └── artifacts/
│   │       ├── sentiment_model.joblib
│   │       ├── evaluation_metrics.json
│   │       └── EVALUATION_REPORT.md
│   ├── pipeline/                  # Review-analysis and comparison logic
│   ├── services/                  # Supporting services, including vision
│   ├── data/                      # Dataset files
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── src/                       # React application source
│   ├── public/
│   ├── package.json
│   └── .env.example
├── supabase/
│   └── migrations/                # Database migrations and RLS policies
├── .env.example                   # Backend configuration template
├── DEPLOYMENT.md                  # Deployment instructions
├── split_csv.py                   # CSV splitting utility
├── render.yaml
├── vercel.json
└── README.md
```

---

## Getting Started

### Prerequisites

Install the following software before running the application:

- Python 3.11 or newer.
- Node.js and npm compatible with the frontend's Vite version.
- Git.
- A Supabase project for authentication.
- Ollama, if you want to use local image defect analysis.

For local model inference, download `gemma3:4b` as described in the [Ollama section](#local-ai-with-ollama-gemma-3-4b).

### 1. Clone the repository

```bash
git clone https://github.com/vikas-g-10/InsightForge.git
cd InsightForge
```

### 2. Configure backend environment variables

Create a root `.env` file from the example.

**Windows Command Prompt:**

```cmd
copy .env.example .env
```

**Linux or macOS:**

```bash
cp .env.example .env
```

Open `.env` and configure the required values for your Supabase project and local environment.

### 3. Configure Supabase

1. Create a project at [Supabase](https://supabase.com).
2. Open **Project Settings → API** and obtain the project URL and public/anonymous key.
3. Configure the authentication Site URL and Redirect URLs to include your local frontend URL, normally `http://localhost:5173/`.
4. Open the Supabase SQL Editor.
5. Run the migration in `supabase/migrations/20261011_create_analysis_history.sql`.

The migration creates the analysis-history table and its Row Level Security policies.

Use the appropriate values in the environment files. Never commit real secrets or private credentials.

### 4. Configure the frontend

Create `frontend/.env.local` from the frontend example.

**Windows Command Prompt:**

```cmd
copy frontend\.env.example frontend\.env.local
```

**Linux or macOS:**

```bash
cp frontend/.env.example frontend/.env.local
```

Edit `frontend/.env.local`:

```env
VITE_API_BASE_URL=
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-supabase-public-anon-key
```

Replace the example Supabase values with your actual project values. Leave `VITE_API_BASE_URL` blank for local development when using the Vite proxy; configure it with your backend URL when connecting to a separately hosted backend.

The Supabase public/anonymous key is intended for the browser client. Never put a service-role key or server-side signing secret in a `VITE_` variable.

---

## Environment Configuration

The root `.env.example` contains the backend configuration template.

| Variable | Purpose |
|---|---|
| `APP_MODE` | Application runtime mode; the default configuration uses `lite` |
| `ALLOWED_ORIGINS` | Comma-separated frontend origins allowed by the backend |
| `MAX_UPLOAD_BYTES` | Maximum review-file upload size; the example sets 50 MiB |
| `MAX_ROWS` | Maximum review rows accepted by the configured ingestion pipeline; the example sets 2,000 |
| `OLLAMA_BASE_URL` | Ollama API base URL |
| `OLLAMA_MODEL` | Vision model name, default `gemma3:4b` |
| `OLLAMA_TIMEOUT_SECONDS` | Vision request timeout |
| `MAX_IMAGE_UPLOAD_BYTES` | Maximum image upload size |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_JWT_AUDIENCE` | Expected JWT audience, normally `authenticated` |
| `AUTH_ENABLED` | Enables protected API authentication |

The default upload configuration allows files up to 50 MiB but limits ingestion to 2,000 data rows. **File size and row count are separate limits.** If you need to process more rows, review the configured `MAX_ROWS` value and any other relevant application or hosting limits before changing it.

---

## Running the Application

Run the backend and frontend in separate terminals.

### Terminal 1: Backend

From the repository root:

```bash
cd backend
```

Create and activate a virtual environment.

**Windows Command Prompt:**

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000 --env-file ../.env
```

**Linux or macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000 --env-file ../.env
```

When the backend starts, check its health endpoint:

[http://localhost:8000/health](http://localhost:8000/health)

Interactive API documentation is available at:

[http://localhost:8000/docs](http://localhost:8000/docs)

### Terminal 2: Frontend

From the repository root:

```bash
cd frontend
npm ci
npm run dev
```

Open the local URL displayed by Vite, normally:

[http://localhost:5173](http://localhost:5173)

Sign in through the Supabase authentication interface to access protected features.

### Verify local vision support

Ensure Ollama is running and `gemma3:4b` has been downloaded. Then open the application's Defect Vision feature or check the authenticated vision-health endpoint.

If the backend is running remotely, make sure `OLLAMA_BASE_URL` points to an Ollama service the backend can actually reach.

---

## Training the Sentiment Model

The repository includes a reproducible training script. The already-generated model artifact can be used for inference without retraining, provided the artifact and its evaluation metrics are present.

To regenerate the dataset and train the models, use a terminal from the repository root.

### 1. Prepare the Python environment

Activate the backend virtual environment and install the dependencies:

```bash
cd backend
python -m venv .venv
```

Activate it using the instructions for your operating system in the previous section, then run:

```bash
pip install -r requirements.txt
```

### 2. Acquire the curated dataset

```bash
python -m ml.dataset --samples-per-class 10000
```

This requests a balanced dataset with 10,000 samples per sentiment class. Dataset acquisition requires internet access.

### 3. Train and evaluate the candidate models

```bash
python -m ml.train
```

The training script creates or updates the model artifacts in `backend/ml/artifacts/`.

Review the results in:

- `backend/ml/artifacts/EVALUATION_REPORT.md`
- `backend/ml/artifacts/evaluation_metrics.json`

Training is optional for normal application use when the saved model artifact already exists.

---

## API Overview

The FastAPI backend exposes endpoints for review ingestion, analysis, comparison, image diagnosis, and machine-learning inference.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Backend health status |
| `POST` | `/api/ingest/validate` | Validate a CSV/XLSX review file |
| `POST` | `/api/analyze` | Analyze review data |
| `POST` | `/api/analyze/file` | Upload and analyze a review file |
| `POST` | `/api/issues` | Retrieve prioritized issue insights |
| `POST` | `/api/simulate` | Simulate the impact of selected fixes |
| `POST` | `/api/emerging-issues` | Detect emerging issue patterns |
| `GET` | `/api/vision/health` | Check Ollama vision availability |
| `POST` | `/api/vision/analyze` | Analyze a product image |
| `POST` | `/api/root-cause/hypotheses` | Generate possible root-cause hypotheses |
| `POST` | `/api/compare/targets` | Discover products and versions available for comparison |
| `POST` | `/api/compare/versions` | Compare two versions of a product |
| `POST` | `/api/compare/products` | Compare two different products |
| `GET` | `/api/ml/metrics` | Retrieve sentiment-model evaluation metrics |
| `POST` | `/api/ml/predict` | Predict sentiment for review text |

Protected endpoints require a valid authentication token when authentication is enabled. Refer to the live API documentation at `/docs` for request schemas and response details.

---

## Deployment

InsightForge separates the frontend from the Python backend.

- **Frontend:** Deploy the React/Vite application to [Vercel](https://vercel.com).
- **Backend:** Deploy the FastAPI application to [Render](https://render.com) or another suitable Python ASGI/container hosting service.
- **Authentication and database:** Configure Supabase.
- **Optional vision inference:** Run Ollama locally for development or configure a securely reachable Ollama service for deployment.

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for detailed deployment settings and environment variables.

**Note:** The cloud backend cannot reach the Ollama process running on your own computer through its own `localhost` address. Vision functionality needs a reachable Ollama instance; the core review-analysis features can still operate without it.

---

## Limitations

- Sentiment labels used for training are derived from star ratings and can contain label noise.
- Neutral and mixed-sentiment reviews are more difficult to classify accurately.
- The trained model was evaluated on an Appliances review subset, so performance may differ on other product categories or writing styles.
- Aspect detection, language identification, and safety screening include heuristic or rule-based components and may make mistakes.
- PII redaction is best-effort and does not guarantee complete anonymization.
- Root-cause outputs are hypotheses requiring verification, not confirmed engineering diagnoses.
- Simulation outputs are estimates, not guarantees of actual rating improvements.
- Ollama vision inference requires an accessible, compatible vision model and sufficient compute resources.
- Deployment providers may impose their own request-size, execution-time, and resource limits.

---

## License

No license has been specified in this repository yet. Unless a license is added, reuse and redistribution are subject to the applicable copyright rules.

---

**InsightForge — Turn customer feedback into product intelligence.**
