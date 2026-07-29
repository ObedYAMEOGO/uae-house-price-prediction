# UAE Rent Prediction

This project is an end-to-end machine learning project that predicts annual property rent (in AED) for residential listings across the UAE — from raw data to a live, deployed web app.

**Live demo:** [https://uae-house-price-prediction-dmkn.vercel.app]; **Model source:** [https://huggingface.co/thehatbuddy]


---

## What this project does

Given a property's basic details like bedrooms, bathrooms, size, city, furnishing status, and neighborhood, the app returns an instant estimate of annual rent in AED.

**Dataset Used:** [https://www.kaggle.com/datasets/alexefimik/dubai-real-estate-transactions-dataset]

Under the hood, it's a Random Forest Regressor trained on ~74,000 real UAE property listings, wrapped in a full production-style pipeline: data cleaning, feature engineering, experiment tracking, and a deployment pipeline.

This isn't a notebook that stops at "here's my accuracy score." It's built the way a real ML system would need to work, with the debugging scars to prove it (more on that below).

<img width="2257" height="1260" alt="pipeline" src="https://github.com/user-attachments/assets/2c43e430-fce6-4b5d-9b18-f68cb6e026c9" />


I hosted the model on Hugging Face ([https://huggingface.co/thehatbuddy](https://huggingface.co/thehatbuddy)) because it was too large to bundle with the application, and the backend downloads it at runtime when needed.

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
   - **A real geocoding bug**: 140 listings tagged to a Sharjah community were geocoded to coordinates in Yemen. Ultimately resolved by dropping `Latitude`/`Longitude` entirely, since categorical location information was more reliable for this dataset.
   - **Severe target skew** (skewness of 83.7) — addressed with a `log1p` transform on `Rent`, reversed at prediction time with `expm1`.
   - **Long-tail categorical cardinality** — 441 unique `Location` values, but the top 108 (≥100 listings each) cover ~91% of the data. Rare locations bucketed into `"Other"` rather than one-hot encoding everything.

2. **Cleaning & feature engineering** — Built with the Strategy, Factory, and Template Method design patterns rather than throwaway scripts, so each step (outlier handling, encoding, bucketing) can be tested and reused.

3. **Model training** — Random Forest Regressor, selected over a linear baseline for its ability to capture non-linear interactions between location, property type, and size. Config-driven via `configs/model_config.yaml`.

4. **Evaluation** — Metrics computed in *both* log-space (fair comparison across runs) and real AED-space (human-readable). Final model: **R² ≈ 0.81 (AED-space)**, **MAE ≈ 25,600 AED**.

5. **Deployment** — A conditional deployment trigger only promotes a model if it clears an MAE threshold, rather than deploying unconditionally on every run — a small but real safeguard against promoting worse models.

---

## Deployment architecture

The trained model (~54MB) needed to be reachable at inference time by a separate FastAPI backend. Three approaches were tried, in this order, for reasons worth documenting honestly:

1. **ZenML's local MLflow model deployer** — works, and is a legitimate way to demonstrate a continuous-deployment pattern (train → evaluate → conditionally deploy → serve), but its local daemon approach is more suitable for demos than production deployments.

2. **Git LFS in the main repo** — committing `model.pkl` via Git LFS worked for local development, but Railway's build process pulled the **LFS pointer file** (a 133-byte text reference) instead of the binary during runtime, causing the runtime to fail to load the model.

3. **Runtime download from Hugging Face (current approach)** — the backend downloads `model.pkl` and `feature_engineer.pkl` from a stable Hugging Face URL on first request, with a size-check guard to avoid partial downloads.

**Practical consequence of this design**: retraining the model locally does *not* automatically update what's live. The Hugging Face Space's copy must be manually updated (`cp` the new artifacts into the Space and push), or the deployment pipeline replaced with a registry-backed workflow.

---

## ZenML Setup Guide

If you've cloned this project and want to run the training pipeline locally, follow these steps to initialize ZenML and connect it to MLflow for experiment tracking.

### Prerequisites

Ensure you have Python 3.8+ and the project dependencies installed:

```bash
git clone https://github.com/ObedYAMEOGO/uae-house-price-prediction.git
cd uae-house-price-prediction
python3 -m venv .venv
# macOS / Linux
source .venv/bin/activate
# On Windows (PowerShell)
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notes:
- If you're on Windows and prefer bash-like behavior, consider Git Bash or WSL for smoother compatibility with some tooling.
- If ZenML or MLflow integrations fail to import, install the integration extras below.

### Install ZenML with MLflow integration (if not already in requirements)

```bash
pip install --upgrade pip
pip install "zenml[mlflow]" mlflow
# optionally install all ZenML integrations (larger):
# pip install "zenml[all]"
zenml integration install mlflow
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

Register an MLflow experiment tracker that will log training metrics and model metadata. This example uses a local SQLite backend file `mlruns.db`.

```bash
zenml experiment-tracker register uae_house_price_mlflow_tracker \
  --flavor=mlflow \
  --tracking_uri="sqlite:///$(pwd)/mlruns.db"
```

On Windows PowerShell you can register with:

```powershell
zenml experiment-tracker register uae_house_price_mlflow_tracker `
  --flavor=mlflow `
  --tracking_uri="sqlite:///$(pwd)/mlruns.db"
```

Notes:
- If you prefer the classic MLflow `mlruns/` directory layout instead of a SQLite file, set tracking_uri to a folder or use a real MLflow server: `--tracking_uri="file:$(pwd)/mlruns"` or set up a remote MLflow server.
- Make sure `mlflow` is installed in your environment (pip install mlflow).

### Optional: Register local artifact store, metadata store, and orchestrator

Registering explicit components makes your stack reproducible and easy to inspect.

```bash
# local artifact store (files on disk)
zenml artifact-store register local_artifact_store --flavor=local --path=./zenml_artifacts

# local SQLite metadata store (ZenML metadata)
zenml metadata-store register sqlite_metadata_store --flavor=sqlite --database="sqlite:///$(pwd)/zenml_metadata.db"

# local orchestrator
zenml orchestrator register local_orchestrator --flavor=local
```

### Step 5: Register MLflow Model Deployer (Optional)

If you want to use ZenML's deployment capabilities:

```bash
zenml model-deployer register uae_house_price_mlflow_deployer --flavor=mlflow
```

### Step 6: Create and set a custom stack

Register a new ZenML stack that combines the components you registered above and make it the active stack:

```bash
zenml stack register uae_house_price_stack \
  -a local_artifact_store \
  -o local_orchestrator \
  -d uae_house_price_mlflow_deployer \
  -e uae_house_price_mlflow_tracker \
  --set
```

If you prefer to reuse the `default` artifact store/orchestrator, replace `local_artifact_store` / `local_orchestrator` with `default`.

Flags explained:
- `-a` — artifact store
- `-o` — orchestrator
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

While your pipeline runs, view training metrics in the MLflow UI.

If you used the SQLite `mlruns.db` backend above:

```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlruns.db
```

If you used the `mlruns/` folder layout instead:

```bash
mlflow ui --backend-store-uri file:$(pwd)/mlruns
```

Then open your browser to `http://localhost:5000` to see experiments, metrics, and model runs.

### Step 9: Run the Training Pipeline

Execute the training pipeline, which will automatically log metrics to MLflow:

```bash
python run_pipeline.py
```

ZenML will orchestrate the pipeline stages (data ingestion → feature engineering → training → evaluation) and log results to your registered MLflow tracker.

You can also list registered pipelines and run a specific pipeline via ZenML CLI:

```bash
zenml pipeline list
# then, to run a pipeline (example)
python pipelines/train_pipeline.py
```

### Ensure MLflow `mlruns` is ignored before pushing to GitHub

If you have an `mlruns/` folder (the MLflow default) or `mlruns.db` from using a SQLite backend, make sure those artifacts are ignored and not committed to the repository. The repository's `.gitignore` already includes `mlruns/` and `mlruns.db`, but if you previously committed them you'll need to untrack them locally before pushing:

```bash
# add to .gitignore if missing (safe even if already present)
echo "mlruns/" >> .gitignore
echo "mlruns.db" >> .gitignore

# If mlruns or mlruns.db were already committed, untrack them while keeping local files
git rm -r --cached mlruns || true
git rm --cached mlruns.db || true

# commit the change to .gitignore and the removal from the index
git add .gitignore
git commit -m "Ignore MLflow tracking artifacts (mlruns/ and mlruns.db)"
git push
```

On Windows PowerShell use Add-Content to append to .gitignore:

```powershell
Add-Content .gitignore "mlruns/"
Add-Content .gitignore "mlruns.db"
```

Notes:
- `git rm --cached` removes files from the Git index (stops tracking) but keeps them on your local disk. This is the safe way to stop tracking MLflow artifacts.

---

## Useful ZenML Commands

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

After running the pipeline, launch the MLflow UI (see Step 8 above) and navigate to `http://localhost:5000` in your browser.

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

**Streamlit would have been the pragmatic choice.** One Python file, no separate frontend framework, no CORS configuration, no cross-origin deployment coordination, deployable to Streamlit Community Cloud. Next.js was chosen anyway because I enjoy building and designing real interfaces, and wanted the practice of shipping a proper separated frontend/backend architecture — the kind of setup a production team might use.

---

## What's next / what others could add

This project stops at "a working, deployed prediction service." Several natural extensions were left out deliberately, either for scope or time:

- **Observability with [Evidently AI](https://www.evidentlyai.com/)** — the single most valuable addition for anyone extending this project. Currently, if the UAE rental market shifts (new developments, policy changes), there is no automated drift detection.
- **A proper model registry with versioned promotion** — rather than manually copying `.pkl` files between three separate locations (local machine, Hugging Face Space, Railway's runtime downloads).
- **Automated retraining on new data** — a scheduled pipeline re-running ingestion → cleaning → training → evaluation → conditional promotion, rather than a manual `python run_pipeline.py` invocation.
- **A/B testing between model versions** — training both a Random Forest and a linear regression baseline (the codebase already supports switching between them via `configs/model_config.yaml`).
- **Rate limiting and stricter CORS** — the deployed API currently allows requests from any origin (`allow_origins=["*"]`), reasonable for a portfolio demo but worth tightening to a specific frontend origin.
- **Automated tests** — the project has none yet. Given the design-pattern-based structure (`Strategy`, `Factory`, `Template Method` classes throughout `src/`), each strategy is independently unit-testable.

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

A meaningful share of the actual work on this project was debugging environment and infrastructure issues rather than modeling: WSL/venv path confusion, ZenML's local daemon flakiness, Git LFS pointer issues, and deployment trade-offs.

---

## License

MIT

**Project Live demo:** [https://uae-house-price-prediction-dmkn.vercel.app]

I hope this project can teach more if you are just starting in ML. I specially worked on this project to fill a gap. How to actually build and ship a real-world Machine learning project?

You can contact me either from here or from my website for any query or collaboration.

**Website:** [https://thehatbuddyai.space]
