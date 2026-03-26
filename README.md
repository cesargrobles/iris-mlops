# Iris Species Classification — Automated Flower Identification

**Course:** MLOps: Master in Business Analytics and Data Science
**Status:** Production-Ready (Modularized Pipeline with CI/CD and Cloud Deployment)
**Group:** Group 3
- Andrea Sabatés
- Tina Jannasch
- Martí Solà
- Ricardo Velásquez
- César González

---

## 1. Business Case

### The Problem

BloomCo's procurement team manually inspects every incoming flower batch to verify species before accepting delivery from suppliers. Staff rely on visual and tactile checks that require botanical expertise most procurement employees don't have. The result: an **88% identification accuracy** that routinely causes downstream problems — wrong species shipped to florist clients, costly returns, and supplier disputes that take days to resolve.

The core issue is not just accuracy. It is **operational fragility**: the business relies on the knowledge and availability of a small number of specialists, and there is no audit trail when a misidentification occurs.

### Why a Model Alone Is Not Enough

A one-off notebook that classifies Iris species would improve accuracy, but it would not solve the operational problem. If the notebook breaks, only the person who wrote it can fix it. If a supplier adds a new column format, the notebook silently produces wrong results. If the specialist is on leave, the business is stuck.

This project delivers a **production-ready ML pipeline**, not a script. The difference matters to the business:

- **Reliability:** Automated tests block broken code from ever reaching production. The CI pipeline validates every change before it can affect live predictions.
- **Trust:** Every training run is logged in Weights & Biases with full metrics, parameters, and the model artifact version. The business can always answer: *which model is live, when was it deployed, and what accuracy did it have at deployment time?*
- **Risk reduction:** The model is versioned and promoted through a registry before serving. Rolling back to a previous version takes seconds, not a debugging session.
- **Maintainability:** Each pipeline stage is a separate, testable module. Changing the feature engineering or swapping the algorithm does not require touching the API or the deployment configuration. Any team member — not just the original author — can extend or fix the system.
- **Delivery speed:** Because the environment is locked (`conda-lock.yml`) and deployment is automated (GitHub Release → Render), moving from a validated experiment to a live API is a single button press.

### What This Delivers for BloomCo

| Before | After |
|---|---|
| Manual inspection, 88% accuracy | Automated prediction, ≥95% accuracy |
| Specialist knowledge required at goods-receipt | Any procurement staff member can operate the tool |
| No audit trail when mislabelling occurs | Full experiment and deployment history in W&B |
| Broken analysis = blocked operations | CI/CD ensures broken code never reaches production |
| Estimated €18,000/year in return costs | Target: reduce return costs by >80% |

### In Scope / Out of Scope

**In scope:** A repeatable, modular MLOps pipeline that classifies Iris species from physical measurements and generates a confidence score per prediction, served via a REST API.

**Out of scope:** Automated purchasing decisions, real-time sensor integration, and classification of flower species beyond the three Iris variants in the training dataset.

---

## 2. Success Metrics

- **Business KPI:** Reduce species mislabelling at goods-receipt from 88% baseline to ≥95% model accuracy, targeting €18,000/year reduction in downstream return costs.
- **Technical Metric:** Weighted F1-score and accuracy on the held-out test set.
- **Acceptance Criteria:** No single species class falls below 90% recall — minority classes must not be systematically missed.
- **Operational Criteria:** API response time < 500ms; CI passes on every PR; deployment tied to a formal GitHub Release.

---

## 3. The Data

### Source and unit of analysis

- Classic Iris dataset, sourced via the Seaborn library (originally UCI ML Repository)
- Unit of analysis: one flower sample with four physical measurements

### Dataset snapshot

| Property | Value |
|---|---|
| Total rows | 150 |
| Features | 4 (sepal length, sepal width, petal length, petal width) |
| Target classes | 3 (setosa, versicolor, virginica) |
| Class distribution | Balanced — 50 samples per class (33.3% each) |
| Sepal length range | 4.3–7.9 cm |
| Petal length range | 1.0–6.9 cm |

### Target definition

`species` — the Iris species of the sample: `setosa`, `versicolor`, or `virginica`.

### Data Dictionary

| Feature | Description | Unit |
|---|---|---|
| `sepal_length` | Length of the sepal (outer petal) | cm |
| `sepal_width` | Width of the sepal | cm |
| `petal_length` | Length of the inner petal | cm |
| `petal_width` | Width of the inner petal | cm |
| `species` | Target — Iris species label | categorical |

### Data sensitivity

This dataset contains no personal or commercially sensitive information. In a production deployment, batch measurement records linked to supplier IDs would be treated as confidential business data and must not be committed to public version control.

---

## 4. Model Card

### Algorithm

| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Solver | lbfgs |
| Max iterations | 500 |
| Class weighting | Balanced (adjusts for class imbalance automatically) |
| Preprocessing | Quantile binning — 3 bins — applied to all 4 features |
| Pipeline artifact | Preprocessor + classifier bundled as a single `.joblib` file |

### Training configuration

| Split | Size | Rows (approx.) |
|---|---|---|
| Training | 70% | 105 |
| Validation | 20% | 30 |
| Test (held-out) | 10% | 15 |

Splits are **stratified** to preserve class proportions. Feature transformations are fitted on training data only — no leakage.

### Expected performance

| Metric | Target |
|---|---|
| Accuracy (validation) | ≥ 95% |
| Weighted F1-score | ≥ 0.95 |
| Per-class recall | ≥ 90% for all three classes |

### Known limitations

- Trained on a balanced, small dataset (150 samples). Real-world performance on heavily imbalanced supplier batches has not been validated.
- Input features must be within the observed training range. Measurements well outside the range (e.g., sensors with systematic bias) may degrade accuracy silently.
- The model classifies only the three Iris species present in the training data. It has no mechanism to flag out-of-distribution samples.
- No data drift detection is implemented in the current version. Model performance should be monitored and the artifact re-promoted if accuracy degrades.

### Model registry

The production model artifact is stored in Weights & Biases under the alias `prod`. The API downloads this artifact at startup — it never loads a local unmanaged file.

---

## 5. MLOps Pipeline Architecture

This project transitions from a fragile Jupyter notebook to a production-grade ML system. Each concern is isolated in its own testable module.

### Core principles

- **Separation of concerns:** Every stage (load, clean, validate, feature engineer, train, evaluate, infer) lives in a dedicated module.
- **Fail-fast validation:** `validate.py` blocks schema violations and missing values before any compute begins.
- **Leakage prevention:** Data is split *before* any feature transformation is fitted.
- **Deployable artifacts:** Preprocessor and classifier are saved as a single `.joblib` pipeline, eliminating training-serving skew.
- **Reproducibility:** Environment locked with `conda-lock.yml` for consistent dependencies across local, CI, and production.
- **Observability:** Structured logging via `logger.py` (console + file) and full experiment tracking with Weights & Biases.
- **Testability:** All modules covered by pytest; CI blocks merges when tests fail.
- **Containerization:** Docker build uses the locked environment for a consistent runtime.
- **CI/CD:** Automated testing on every PR; deployment triggered only by a formal GitHub Release.

### Pipeline stages

1. **`load_data.py`** — Fetches raw data from source
2. **`clean_data.py`** — Handles missing values and standardises column names
3. **`validate.py`** — Enforces schema constraints and data quality gates
4. **`features.py`** — Builds the sklearn `ColumnTransformer` preprocessing recipe
5. **`train.py`** — Fits Logistic Regression inside a sklearn `Pipeline`
6. **`evaluate.py`** — Computes metrics and logs to W&B
7. **`infer.py`** — Generates predictions with confidence scores
8. **`api.py`** — FastAPI application for real-time serving (downloads model from W&B registry)

### Configuration management

| Layer | Purpose |
|---|---|
| `config.yaml` | All non-secret runtime settings (paths, model params, split sizes) |
| `.env` | Secrets only (W&B API key) — never committed |
| CLI args | Optional runtime overrides via `--config` flag |

---

## 6. Repository Structure

```text
.
├── .env.example                    # Environment variables template
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Continuous Integration — runs on every PR
│       └── deploy.yml              # Deployment — triggered by GitHub Release
├── config.yaml                     # All non-secret pipeline configuration
├── conda-lock.yml                  # Locked environment for reproducibility
├── Dockerfile                      # Container definition (uses conda-lock)
├── .dockerignore                   # Excludes dev artifacts from image
├── environment.yml                 # Conda environment specification
├── pytest.ini                      # Test configuration
├── requirements.txt                # Python dependencies (fallback)
├── data/
│   ├── raw/iris.csv                # Raw input data
│   └── processed/clean.csv        # Cleaned data artifact
├── logs/app.log                    # Structured application logs
├── models/model.joblib             # Trained model artifact (local cache)
├── notebooks/
│   ├── 00_iris_analysis_vLegacy.ipynb  # Original exploratory notebook
│   └── 01_iris_analysis_vExp.ipynb     # Modular sandbox (reads from src/)
├── reports/predictions.csv        # Inference output
├── src/
│   ├── api.py                      # FastAPI serving layer
│   ├── clean_data.py               # Data cleaning logic
│   ├── config.py                   # Configuration loader
│   ├── evaluate.py                 # Model evaluation and W&B logging
│   ├── features.py                 # Feature engineering pipeline
│   ├── infer.py                    # Inference logic
│   ├── load_data.py                # Data loading
│   ├── logger.py                   # Structured logging (console + file)
│   ├── main.py                     # Pipeline orchestrator entry point
│   ├── train.py                    # Model training
│   ├── utils.py                    # Artifact save/load helpers
│   └── validate.py                 # Data validation gates
└── tests/
    ├── test_clean_data.py
    ├── test_evaluate.py
    ├── test_features.py
    ├── test_infer.py
    ├── test_load_data.py
    ├── test_main.py
    ├── test_train.py
    ├── test_utils.py
    └── test_validate.py
```

---

## 7. Setup and Installation

### Prerequisites

- Python 3.11+
- Conda / Miniconda
- Docker (for containerised deployment)
- A Weights & Biases account (free tier is sufficient)

### Local development setup

**1. Clone the repository:**
```bash
git clone https://github.com/cesargrobles/iris-mlops.git
cd iris-mlops
```

**2. Set up the environment:**
```bash
# Recommended: use the locked environment for full reproducibility
conda install -c conda-forge conda-lock
conda-lock install --name mlops-kickoff-2 conda-lock.yml
conda activate mlops-kickoff-2

# Alternative: create from environment.yml
conda env create -f environment.yml
conda activate mlops-kickoff-2
```

**3. Configure secrets:**
```bash
cp .env.example .env
# Edit .env and add your WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT
```

---

## 8. Usage

### Run the full training pipeline

```bash
python -m src.main --config config.yaml
```

This runs: load → clean → validate → split → train → evaluate → save → infer → persist predictions.

### Run tests

```bash
python -m pytest -q
```

### Start the local API server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

**Health check:**
```bash
curl http://localhost:8080/health
```

**Make a prediction:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"sepal_length":5.0,"sepal_width":3.5,"petal_length":1.3,"petal_width":0.3}]}' \
  http://localhost:8080/predict
```

---

## 9. Docker Deployment

```bash
# Build
docker build -t iris-mlops .

# Run locally
docker run -p 8080:8080 --env-file .env iris-mlops

# Verify
curl http://localhost:8080/health
```

The Docker image uses the `conda-lock.yml` for exact dependency pinning. Development artifacts (`tests/`, `notebooks/`, `data/`, `reports/`, `wandb/`, `.github/`) are excluded via `.dockerignore`.

---

## 10. Weights & Biases Integration

W&B is used for experiment tracking, artifact management, and the model registry.

### Configuration

Set the following in `config.yaml`:
```yaml
wandb:
  enabled: true
  project: "session1-kickoff-prueba"
  entity: "cesargrobles-ie-university"
```

And in `.env`:
```
WANDB_API_KEY=your_key_here
WANDB_ENTITY=cesargrobles-ie-university
WANDB_PROJECT=session1-kickoff-prueba
WANDB_MODEL_ARTIFACT=iris_model
```

### Workflow

1. `main.py` logs hyperparameters, metrics, and the trained `.joblib` artifact to W&B
2. The artifact is promoted with the alias `prod` at the end of a successful training run
3. At API startup, `api.py` downloads the `prod` artifact from the W&B registry — it never loads an unmanaged local file
4. All evaluation metrics are tracked across runs for comparison

---

## 11. CI/CD Pipeline

### Continuous Integration (`ci.yml`)

- **Trigger:** Pull requests and pushes to `main`
- **Steps:** Checkout → setup Miniconda → install locked environment → run pytest → validate Docker build
- **Effect:** PRs cannot be merged until all tests pass and the Docker image builds successfully

### Continuous Deployment (`deploy.yml`)

- **Trigger:** Publishing a formal GitHub Release from `main`
- **Steps:** Checkout → install dependencies → run tests → build Docker image → deploy to Render
- **Effect:** Production deployments are always tied to a tagged, reviewed release — no ad-hoc deploys

---

## 12. Cloud Deployment

**Live URL:** https://iris-mlops.onrender.com

The application is deployed on Render as a Docker web service. The container downloads the `prod` model artifact from W&B at startup.

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok", "model_loaded": true}` |
| `/predict` | POST | Accepts a JSON batch, returns predictions with confidence scores |

### Example (live deployment)

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"sepal_length":5.0,"sepal_width":3.5,"petal_length":1.3,"petal_width":0.3}]}' \
  https://iris-mlops.onrender.com/predict
```

---

## 13. Testing Strategy

| Category | Modules covered |
|---|---|
| Data pipeline | `test_load_data.py`, `test_clean_data.py`, `test_validate.py` |
| Feature engineering | `test_features.py` |
| Model training | `test_train.py` |
| Inference | `test_infer.py` |
| Utilities | `test_utils.py` |
| Evaluation | `test_evaluate.py` |
| Integration | `test_main.py` (end-to-end pipeline) |

All tests use synthetic data for determinism and mock external dependencies (W&B) to avoid network calls in CI.

---

## 14. Evaluation Checklist for Reviewers

1. **Clone and setup:** Follow Section 7 above
2. **Run tests:** `python -m pytest -q` — all should pass
3. **Run pipeline:** `python -m src.main --config config.yaml`
4. **Check artifacts:** `models/model.joblib`, `reports/predictions.csv`, `data/processed/clean.csv`
5. **Start API:** `uvicorn src.api:app --host 0.0.0.0 --port 8080`
6. **Test endpoints:** `/health` and `/predict` as shown in Section 8
7. **Docker build:** `docker build -t iris-mlops .`
8. **Live service:** `curl https://iris-mlops.onrender.com/health`

### Key areas to assess

- **Modularity:** Each `src/` module has a single responsibility
- **Configuration:** No hardcoded runtime values — all in `config.yaml` or `.env`
- **Logging:** Zero `print()` in production code; dual-output logger (console + file)
- **Testing:** All modules covered; CI blocks on failures
- **W&B registry:** Training promotes artifact with `prod` alias; API uses `prod` artifact
- **Docker:** Lean image with strict `.dockerignore`; built with `conda-lock.yml`
- **Deployment:** Live Render app responds to real JSON payloads

---

## 15. Changelog

### v1.0.0 — 2026-03-24

- Initial production release
- Full modular pipeline: load → clean → validate → features → train → evaluate → infer
- FastAPI serving layer with Pydantic request validation
- W&B experiment tracking with `prod` artifact alias
- API downloads model from W&B registry at startup (no local unmanaged artifacts)
- Docker deployment using `conda-lock.yml` for exact reproducibility
- CI pipeline validates all PRs; deployment triggered by GitHub Release
- Live deployment on Render: https://iris-mlops.onrender.com
- Comprehensive pytest suite covering all core modules

---

## 16. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
