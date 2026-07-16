# UAE Rent Prediction

This project is an end-to-end machine learning project that predicts annual property rent (in AED) for residential listings across the UAE — from raw data to a live, deployed web app.

**Live demo:** [https://uae-house-price-prediction-dmkn.vercel.app]; **Model source:** [https://huggingface.co/thehatbuddy]

---

## What this project does

Given a property's basic details like bedrooms, bathrooms, size, city, furnishing status, and neighborhood, the app returns an instant estimate of annual rent in AED.

Under the hood, it's a Random Forest Regressor trained on ~74,000 real UAE property listings, wrapped in a full production-style pipeline: data cleaning, feature engineering, experiment tracking, a REST API, and a deployed frontend.

This isn't a notebook that stops at "here's my accuracy score." It's built the way a real ML system would need to work, with the debugging scars to prove it (more on that below).

<img width="3167" height="628" alt="UAE_Rent_Prediction_Architecture" src="https://github.com/user-attachments/assets/59119ece-092d-481f-8cfd-a7dc241abba2" />

I hosted the model on Hugging Face ([https://huggingface.co/thehatbuddy](https://huggingface.co/thehatbuddy)) because it was too large to bundle with the application, and the backend downloads it at container startup instead of baking it into the Docker image or relying on Git LFS in the main repository. This approach avoids Git LFS pointer issues during Railway deployments by ensuring the actual model is fetched directly at runtime, as explained in the [Deployment Architecture](#deployment-architecture) section.

---

## Tech stack

| Layer | Tools |
|---|---|
| Data & EDA | pandas, seaborn, matplotlib |
| Pipeline orchestration | [ZenML](https://zenml.io) |
| Experiment tracking & model registry | [MLflow](https://mlflow.org) |
| Model | scikit-learn (Random Forest Regressor) |
| Backend API | FastAPI, Docker |
| Backend hosting | Railway |
| Model artifact hosting | Hugging Face Spaces |
| Frontend | Next.js, TypeScript, Tailwind CSS |
| Frontend hosting | Vercel |

---

## The pipeline

1. **EDA** — Investigated ~74,000 listings before writing any modeling code. Found and addressed:
   - **Target leakage**: `Rent_per_sqft` was mathematically derived from `Rent / Area_in_sqft` — an exact leak, confirmed numerically before dropping it.
   - **A real geocoding bug**: 140 listings tagged to a Sharjah community were geocoded to coordinates in Yemen. Ultimately resolved by dropping `Latitude`/`Longitude` entirely, since categorical `City`/`Location` already carried strong location signal.
   - **Severe target skew** (skewness of 83.7) — addressed with a `log1p` transform on `Rent`, reversed at prediction time with `expm1`.
   - **Long-tail categorical cardinality** — 441 unique `Location` values, but the top 108 (≥100 listings each) cover ~91% of the data. Rare locations bucketed into `"Other"` rather than one-hot-encoding all 441.

2. **Cleaning & feature engineering** — Built with the Strategy, Factory, and Template Method design patterns rather than throwaway scripts, so each step (outlier handling, encoding, bucketing) is independently swappable and testable. Outlier removal uses a **percentile-based** method (not Z-score or IQR), deliberately chosen because the target's severe skew makes those methods misclassify legitimate high-value listings as outliers.

3. **Model training** — Random Forest Regressor, selected over a linear baseline for its ability to capture non-linear interactions between location, property type, and size. Config-driven via `configs/model_config.yaml`, so switching models or tuning hyperparameters doesn't require touching code.

4. **Evaluation** — Metrics computed in *both* log-space (fair comparison across runs) and real AED-space (human-readable). Final model: **R² ≈ 0.81 (AED-space)**, **MAE ≈ 25,600 AED**.

5. **Deployment** — A conditional deployment trigger only promotes a model if it clears an MAE threshold, rather than deploying unconditionally on every run — a small but real safeguard against silently shipping a worse model.

---

## Deployment architecture

The trained model (~54MB) needed to be reachable at inference time by a separate FastAPI backend. Three approaches were tried, in this order, for reasons worth documenting honestly:

1. **ZenML's local MLflow model deployer** — works, and is a legitimate way to demonstrate a continuous-deployment pattern (train → evaluate → conditionally deploy → serve), but its local daemon process proved unreliable in a WSL environment specifically — it would die between separate script invocations, requiring manual restarts. Kept in the codebase (`pipelines/deployment_pipeline.py`) as a demonstration of the pattern, but not used for the actual live deployment.

2. **Git LFS in the main repo** — committing `model.pkl` via Git LFS worked for local development, but Railway's build process pulled the **LFS pointer file** (a 133-byte text reference) instead of resolving it to the real binary, since its build environment doesn't automatically run `git lfs pull` during checkout. This silently broke model loading in production (`model_loaded: false`) despite working perfectly locally.

3. **Runtime download from Hugging Face (current approach)** — the backend downloads `model.pkl` and `feature_engineer.pkl` from a stable Hugging Face URL on first request, with a size-check guard against ever loading a stale pointer file again. This decouples the backend's deployment from any git/LFS quirks entirely, and reuses infrastructure (the Hugging Face Space) already proven reliable.

**Practical consequence of this design**: retraining the model locally does *not* automatically update what's live. The Hugging Face Space's copy must be manually updated (`cp` the new artifacts into the Space's local clone, commit, push) before Railway's next cold start will pick up the change. For a project without frequent retraining, this manual sync is an acceptable tradeoff; a production system with active retraining would want a proper model registry with versioned promotion instead.

---

## ZenML Setup Guide

If you've cloned this project and want to run the training pipeline locally, follow these steps to initialize ZenML and connect it to MLflow for experiment tracking.

### Prerequisites

Ensure you have Python 3.8+ and the project dependencies installed:

```bash
git clone https://github.com/ObedYAMEOGO/uae-house-price-prediction.git
cd uae-house-price-prediction
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate but I strongly recommenyou to switch to a linux distribution because it fill fix your daemon flakiness issues. I am using Windows OS, got to install WSL so that I could get access to a linux friendly env to run my project. 
pip install -r requirements.txt
```

### Step 1: Initialize ZenML

Initialize a new ZenML repository in your project:

```bash
zenml init
```

This creates a `.zenml/` directory with default configurations.

### Step 2: Check ZenML Status

Verify the installation and current setup:

```bash
zenml status
```

You should see information about your ZenML environment, Python version, and installed extensions.

### Step 3: List Available Stacks

View all available ZenML stacks (combinations of components like orchestrators, experiment trackers, and model deployers):

```bash
zenml stack list
```

Initially, you'll see a `default` stack.

### Step 4: Register MLflow Experiment Tracker

Register an MLflow experiment tracker that will log training metrics and model metadata:

```bash
zenml experiment-tracker register uae_house_price_mlflow_tracker \
  --flavor=mlflow \
  --tracking_uri="sqlite:///$(pwd)/mlruns.db"
```

This creates a local SQLite database for MLflow (`mlruns.db`) in your project directory. On Windows, use:

```bash
zenml experiment-tracker register uae_house_price_mlflow_tracker ^
  --flavor=mlflow ^
  --tracking_uri="sqlite:///mlruns.db"
```

### Step 5: Register MLflow Model Deployer (Optional)

If you want to use ZenML's deployment capabilities:

```bash
zenml model-deployer register uae_house_price_mlflow_deployer --flavor=mlflow
```

### Step 6: Create a Custom Stack

Register a new ZenML stack that combines default orchestrator, artifact store, and your MLflow components:

```bash
zenml stack register uae_house_price_stack \
  -a default \
  -o default \
  -d uae_house_price_mlflow_deployer \
  -e uae_house_price_mlflow_tracker \
  --set
```
Note that you can name your stack as you want here I choose to call it `uae_house_price_stack`, `uae_house_price_mlflow_deployer`, `uae_house_price_mlflow_tracker`

Flags explained:
- `-a default` — use the default artifact store (local filesystem)
- `-o default` — use the default orchestrator (local)
- `-d` — model deployer (MLflow)
- `-e` — experiment tracker (MLflow)
- `--set` — make this the active stack

### Step 7: Verify Your Stack

Confirm the stack is set correctly:

```bash
zenml stack describe
```

You should see your stack configuration with the registered components.

### Step 8: Launch MLflow UI (for monitoring)

While your pipeline runs, view training metrics in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Then open your browser to `http://localhost:5000` to see experiments, metrics, and model runs.

### Step 9: Run the Training Pipeline

Execute the training pipeline, which will automatically log metrics to MLflow:

```bash
python run_pipeline.py
```

ZenML will orchestrate the pipeline stages (data ingestion → feature engineering → training → evaluation) and log results to your registered MLflow tracker.

### Useful ZenML Commands

```bash
# View all registered stacks
zenml stack list

# Switch to a different stack
zenml stack set <stack_name>

# Inspect a specific stack
zenml stack describe <stack_name>

# List all experiment trackers
zenml experiment-tracker list

# List all model deployers
zenml model-deployer list

# Delete a stack (if needed)
zenml stack delete <stack_name>
```

---

## MLflow Experiment Tracking & Metrics

This project logs all training runs to MLflow for easy comparison and reproducibility.

### Accessing MLflow

After running the pipeline, launch the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Navigate to `http://localhost:5000` in your browser. Refer to the demo below.

### Interpreting the Metrics

- **R² Score**: Proportion of variance explained. 0.81 means the model explains 81% of rent variation.
- **MAE (Mean Absolute Error)**: Average prediction error. 25,600 AED means predictions are off by ~26k AED on average.
- **Log-space metrics**: Used for fair comparison across runs with different data preprocessing; reported separately from human-readable AED-space metrics.

### Comparing Runs in MLflow

1. **View all experiments**: The `Default` experiment collects all pipeline runs.
2. **Compare metrics**: MLflow's compare view shows side-by-side performance across runs (useful when testing hyperparameter changes).
3. **Download artifacts**: Each run's trained model, feature engineer object, and config are saved and downloadable.

### Using Metrics for Model Selection

The deployment pipeline uses the MAE threshold to decide whether to promote a model:

```python
if mae_aed_space < DEPLOYMENT_MAE_THRESHOLD:
    # Promote to production
    deploy_model()
else:
    # Log warning and skip deployment
    logger.warning(f"MAE {mae_aed_space} exceeds threshold")
```

This ensures only models meeting quality standards are deployed.

---

## Why Next.js instead of Streamlit?

Honest answer: personal preference, not necessity.

**Streamlit would have been the pragmatic choice.** One Python file, no separate frontend framework, no CORS configuration, no cross-origin deployment coordination, deployable to Streamlit Community Cloud in about fifteen minutes. For anyone wanting to get a model in front of people with minimal friction, Streamlit (or Gradio) is genuinely the right tool, and this project's own deployment journey — Hugging Face quota bugs, Railway build-context mismatches, Git LFS pointer resolution failures — is a fairly strong argument *for* the simpler path.

Next.js was chosen anyway because I enjoy building and designing real interfaces, and wanted the practice of shipping a proper separated frontend/backend architecture — the kind of setup a production web app would actually use, with its own design system rather than an auto-generated widget layout. That's a legitimate reason to pick the harder path for a portfolio project, but it isn't the *only* reasonable choice, and I'd point a beginner toward Streamlit first if their goal is simply "get a model into a usable demo, fast."

---

## What's next / what others could add

This project stops at "a working, deployed prediction service." Several natural extensions were left out deliberately, either for scope or time:

- **Observability with [Evidently AI](https://www.evidentlyai.com/)** — the single most valuable addition for anyone extending this project. Currently, if the UAE rental market shifts (new developments, pricing changes, seasonal effects), the deployed model has no way of knowing its predictions are drifting from reality. Evidently could be wired in as a scheduled ZenML pipeline that:
  - Compares incoming prediction requests against the training data's feature distributions (data drift detection)
  - Flags when categorical distributions shift (e.g., a sudden influx of requests for locations barely represented in training)
  - Generates a dashboard report reviewable on a cadence, rather than silently trusting a model that may be stale
  
  This is the difference between "I trained a model" and "I built a system that knows when it needs retraining" — a meaningfully more advanced signal for anyone reviewing the project.

- **A proper model registry with versioned promotion** — rather than manually copying `.pkl` files between three separate locations (local machine, Hugging Face Space, Railway's runtime download), a registry (MLflow's own, or a cloud alternative) with a "Production" alias would let the backend always pull the latest promoted version automatically.

- **Automated retraining on new data** — a scheduled pipeline re-running ingestion → cleaning → training → evaluation → conditional promotion, rather than a manual `python run_pipeline.py` invocation.

- **A/B testing between model versions** — training both a Random Forest and a linear regression baseline (the codebase already supports switching between them via `configs/model_config.yaml`) and logging both to MLflow for side-by-side comparison, or serving both simultaneously to compare real-world performance.

- **Rate limiting and stricter CORS** — the deployed API currently allows requests from any origin (`allow_origins=["*"]`), reasonable for a portfolio demo but worth tightening (to the actual frontend domain) and adding basic rate limiting before treating this as anything beyond a demo.

- **Automated tests** — the project has none yet. Given the design-pattern-based structure (`Strategy`, `Factory`, `Template Method` classes throughout `src/`), each strategy is independently unit-testable in isolation — a natural next step for anyone picking this up.

---

## Local setup

```bash
git clone https://github.com/ObedYAMEOGO/uae-house-price-prediction.git
cd uae-house-price-prediction

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# download the dataset manually from Kaggle into data/raw/archive.zip, then:
python run_pipeline.py

# backend
uvicorn backend.app:app --reload --port 8080

# frontend (separate terminal)
cd frontend
npm install
npm run dev
```

---

## Docker Setup (Production)

Build and run with Docker Compose locally:

```bash
docker-compose up
```

This spins up:
- **Backend** on `http://localhost:8080`
- **Frontend** on `http://localhost:3000`

Models load from local storage or download from Hugging Face on first request.

---

## A note on the process

A meaningful share of the actual work on this project was debugging environment and infrastructure issues rather than modeling: WSL/venv path confusion, ZenML's local daemon flakiness, Git LFS pointer resolution failing silently in a CI environment, a Hugging Face Spaces quota bug affecting multiple users, and several Docker build-context mismatches between local, Hugging Face, and Railway. None of that is a footnote — it's genuinely most of what building and deploying a real ML project looks like, and it's the part most tutorials skip. If you're newer to this kind of project, expect that ratio, and don't take it as a sign something's wrong with your approach.

---

## License

MIT

**Project ive demo:** [https://uae-house-price-prediction-dmkn.vercel.app]

I hope this project can teach more if you are just starting in ML. 

You can contact me either from here or from my website for any query or collaboration.

**Website:** [https://thehatbuddyai.space]
