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

## 1. Business Objective

Wholesale florists currently rely on manual botanical inspection to verify the species of flower batches at goods-receipt. This project automates that verification step using physical measurements.

* **The Goal:** Predict the species of an Iris flower (Setosa, Versicolor, or Virginica) from four physical measurements taken at the point of goods-receipt, replacing a slow and error-prone manual process.
* **The User:** Procurement staff at BloomCo who need to verify incoming flower batches quickly and accurately without specialist botanical knowledge.
* **In Scope:** A repeatable, modular MLOps pipeline that classifies Iris species and generates a confidence score per prediction.
* **Out of Scope:** Automated purchasing decisions, real-time sensor integration, or classification of flower species beyond the three Iris variants in the dataset.

---

## 2. Success Metrics

* **Business KPI:** Reduce the species mislabelling rate at goods-receipt from the current baseline of **88% manual accuracy** to a target of **≥ 95% model accuracy**, cutting downstream return costs estimated at €18,000/year.
* **Technical Metric:** **Accuracy** and **F1-Score (weighted)** on the validation set, balancing correct identification across all three species classes.
* **Acceptance Criteria:** The model must perform consistently across all three species — no single class should fall below 90% recall, ensuring minority-class species are not systematically missed.

---

## 3. The Data

### Source and unit of analysis
- The classic Iris dataset, sourced via the Seaborn library
- Unit of analysis is a single flower sample with four physical measurements

### Dataset snapshot
- Rows: 150
- Columns: 5 (4 features + 1 target)
- Class distribution: perfectly balanced — 50 samples per species (33.3% each)
- Feature value ranges: sepal length 4.3–7.9 cm, petal length 1.0–6.9 cm

### Target definition
- `species`: the Iris species of the sample — one of `setosa`, `versicolor`, or `virginica`

### Data sensitivity
- This dataset contains no personal or commercially sensitive information
- In a production deployment, batch measurement records linked to supplier IDs would be treated as confidential business data and must not be committed to public version control

### Data Dictionary

| Feature | Description | Unit |
|---|---|---|
| `species` | Target — Iris species label | categorical |
| `sepal_length` | Length of the sepal (outer petal) | cm |
| `sepal_width` | Width of the sepal | cm |
| `petal_length` | Length of the inner petal | cm |
| `petal_width` | Width of the inner petal | cm |

---

## 4. MLOps Pipeline Architecture

This repository implements a complete **Machine Learning Operations (MLOps)** pipeline, transitioning from fragile Jupyter Notebooks to production-ready, testable software engineering architecture.

### Core Principles
* **Separation of Concerns:** Every step (Loading, Cleaning, Validating, Training, Evaluation, Inference) has a dedicated, single-purpose Python module.
* **Fail-Fast Security Gates:** `validate.py` blocks missing values and schema violations before expensive compute begins.
* **Leakage Prevention:** Data is split *before* fitting any feature transformations or the model.
* **Deployable Artifacts:** The orchestrator bundles preprocessing and the classifier into a single `.joblib` file, preventing training-serving skew.
* **Reproducibility:** Environment is locked with `conda-lock.yml` for consistent dependencies across development, CI, and production.
* **Observability:** Structured logging via `logger.py` and experiment tracking with Weights & Biases (W&B).
* **Testing:** Comprehensive unit tests for all modules with pytest.
* **Containerization:** Docker-based deployment for consistent runtime environment.
* **CI/CD:** Automated testing and deployment pipelines with GitHub Actions.

### Pipeline Stages
1. **Data Loading (`load_data.py`):** Fetches raw data from source
2. **Data Cleaning (`clean_data.py`):** Handles missing values, outliers, and basic transformations
3. **Data Validation (`validate.py`):** Enforces schema constraints and data quality checks
4. **Feature Engineering (`features.py`):** Creates derived features and preprocessing pipeline
5. **Model Training (`train.py`):** Fits the machine learning model
6. **Model Evaluation (`evaluate.py`):** Computes metrics and logs to W&B
7. **Model Inference (`infer.py`):** Generates predictions on new data
8. **API Serving (`api.py`):** FastAPI application for real-time predictions

### Configuration Management
- **Static Config:** `config.yaml` contains all non-secret settings
- **Environment Variables:** Secrets (API keys) loaded from `.env` file
- **Runtime Overrides:** Command-line arguments can override config values

---

## 5. Repository Structure

```text
.
├── .env.example                    # Environment variables template
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Continuous Integration pipeline
│       └── deploy.yml              # Deployment automation
├── config.yaml                     # Pipeline configuration
├── conda-lock.yml                  # Locked environment dependencies
├── Dockerfile                      # Container definition
├── environment.yml                 # Conda environment specification
├── pytest.ini                      # Test configuration
├── requirements.txt                # Python dependencies (fallback)
├── data/
│   ├── raw/                        # Raw input data
│   │   └── iris.csv
│   └── processed/                  # Cleaned data
│       └── clean.csv
├── logs/                           # Application logs
├── models/                         # Trained model artifacts
│   └── model.joblib
├── notebooks/                      # Exploratory analysis
│   ├── 00_iris_analysis_vLegacy.ipynb
│   └── 01_iris_analysis_vExp.ipynb
├── reports/                        # Prediction outputs
│   └── predictions.csv
├── src/                            # Source code
│   ├── __init__.py
│   ├── api.py                      # FastAPI application
│   ├── clean_data.py               # Data cleaning logic
│   ├── config.py                   # Configuration loading
│   ├── evaluate.py                 # Model evaluation
│   ├── features.py                 # Feature engineering
│   ├── infer.py                    # Inference logic
│   ├── load_data.py                # Data loading
│   ├── logger.py                   # Logging configuration
│   ├── main.py                     # Pipeline orchestrator
│   ├── train.py                    # Model training
│   ├── utils.py                    # Utility functions
│   └── validate.py                 # Data validation
└── tests/                          # Unit tests
    ├── __init__.py
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

## 6. Setup and Installation

### Prerequisites
- Python 3.11+
- Conda/Minconda
- Docker (for containerized deployment)
- GitHub account (for CI/CD)

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd iris-mlops
   ```

2. **Environment Setup:**
   ```bash
   # Create conda environment
   conda env create -f environment.yml
   conda activate mlops-kickoff-2

   # Or use locked environment for reproducibility
   conda install -c conda-forge conda-lock
   conda-lock install --name mlops-kickoff-2 conda-lock.yml
   conda activate mlops-kickoff-2
   ```

3. **Configure Environment Variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your W&B credentials
   ```

4. **Install in development mode:**
   ```bash
   pip install -e .
   ```

---

## 7. Usage

### Exploratory Analysis
Launch the interactive notebook for data exploration:
```bash
jupyter notebook notebooks/01_iris_analysis_vExp.ipynb
```

### Run the Full Pipeline
Execute the end-to-end MLOps pipeline:
```bash
python -m src.main --config config.yaml
```

### Run Tests
Execute the test suite:
```bash
python -m pytest -q
```

### Start Local API Server
Serve predictions via REST API:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

Test the API:
```bash
# Health check
curl http://localhost:8080/health

# Make prediction
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"sepal_length":5.0,"sepal_width":3.5,"petal_length":1.3,"petal_width":0.3}]}' \
  http://localhost:8080/predict
```

---

## 8. Docker Deployment

### Build the Container
```bash
docker build -t iris-mlops .
```

### Run Locally
```bash
docker run -p 8080:8080 --env-file .env iris-mlops
```

### Verify Container
```bash
curl http://localhost:8080/health
```

---

## 9. Weights & Biases (W&B) Integration

This project uses W&B for experiment tracking and model versioning:

### Configuration
Add W&B settings to `config.yaml`:
```yaml
wandb:
  enabled: true
  project: "session1-kickoff-prueba"
  entity: "cesargrobles-ie-university"
```

### Features
- **Experiment Logging:** Automatic logging of hyperparameters, metrics, and artifacts
- **Model Registry:** Versioned model storage and retrieval
- **Artifact Management:** Tracking of datasets, models, and evaluation results
- **Production Deployment:** API server downloads latest model from W&B registry

### W&B Workflow
1. Training logs experiments to W&B
2. Model artifacts are uploaded as W&B Artifacts
3. API server downloads model from W&B for inference
4. Evaluation metrics are tracked across runs

---

## 10. CI/CD Pipeline

### Continuous Integration (CI)
- **Trigger:** Pull requests and pushes to `main` branch
- **Environment:** Ubuntu latest with Miniconda
- **Steps:**
  - Checkout code
  - Setup conda environment
  - Run test suite
  - Validate Docker build
- **Configuration:** `.github/workflows/ci.yml`

### Continuous Deployment (CD)
- **Trigger:** Manual dispatch or GitHub releases
- **Environment:** Ubuntu latest
- **Steps:**
  - Checkout code
  - Setup Python environment
  - Run tests
  - Build Docker image
  - Deploy to cloud platform (Render/Heroku/GCP/AWS)
- **Configuration:** `.github/workflows/deploy.yml`

---

## 11. Cloud Deployment

### Render Deployment
The application is deployed on Render with the following configuration:

- **Service Type:** Web Service
- **Runtime:** Docker
- **Build Command:** `docker build -t iris-mlops .`
- **Start Command:** `docker run -p $PORT:8080 --env-file .env iris-mlops`
- **Environment Variables:** Configure W&B credentials in Render dashboard
- **Health Check:** `/health` endpoint

**Live Deployment URL:** https://iris-mlops.onrender.com

### API Endpoints
- `GET /health` - Health check
- `POST /predict` - Batch prediction endpoint

### Example API Usage
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"sepal_length":5.0,"sepal_width":3.5,"petal_length":1.3,"petal_width":0.3}]}' \
  https://iris-mlops.onrender.com/predict
```

---

## 12. Testing Strategy

### Unit Tests
- **Framework:** pytest
- **Coverage:** All core modules tested
- **CI Integration:** Tests run on every PR and push
- **Command:** `python -m pytest -q`

### Test Categories
- **Data Pipeline:** Loading, cleaning, validation
- **Feature Engineering:** Preprocessing pipeline
- **Model Training:** Algorithm fitting and serialization
- **Inference:** Prediction generation
- **Integration:** End-to-end pipeline execution

### Test Data
- Synthetic test data for deterministic testing
- Mock objects for external dependencies (W&B)
- Edge cases and error conditions

---

## 13. Evaluation Instructions for Reviewers

### Quick Start Checklist
1. ✅ **Clone and Setup:** Follow setup instructions above
2. ✅ **Run Tests:** `python -m pytest -q` (should pass 100%)
3. ✅ **Execute Pipeline:** `python -m src.main --config config.yaml`
4. ✅ **Check Outputs:** Verify `models/model.joblib`, `reports/predictions.csv`, and `data/processed/clean.csv` are created
5. ✅ **Test API:** Start server with `uvicorn src.api:app --host 0.0.0.0 --port 8080` and test endpoints
6. ✅ **Docker Build:** `docker build -t iris-mlops .` completes successfully
7. ✅ **Live Deployment:** Visit https://iris-mlops.onrender.com/health

### Key Deliverables to Evaluate
- **Code Quality:** Modular architecture, type hints, documentation
- **Testing:** Comprehensive test coverage, CI passing
- **Configuration:** Proper separation of config and secrets
- **Containerization:** Working Dockerfile with conda-lock
- **CI/CD:** Automated testing and deployment workflows
- **Experiment Tracking:** W&B integration for logging and artifacts
- **API Design:** RESTful endpoints with proper error handling
- **Deployment:** Live service on Render with health checks
- **Documentation:** Clear setup, usage, and architecture explanations

### Performance Benchmarks
- **Model Accuracy:** ≥ 95% on validation set
- **API Response Time:** < 500ms for single prediction
- **Container Size:** < 2GB Docker image
- **Test Coverage:** 100% core functionality

### Common Issues & Troubleshooting
- **W&B Connection:** Ensure `.env` has valid API key
- **Port Conflicts:** Change port if 8080 is occupied
- **Conda Environment:** Use `conda-lock` for exact reproducibility
- **Docker Build:** May take 5-10 minutes on first run

---

## 14. Future Roadmap

### Phase 2: Advanced MLOps
* **Model Registry:** MLflow integration for model versioning
* **Monitoring:** Data drift detection and model performance monitoring
* **A/B Testing:** Multi-model deployment with traffic splitting
* **Feature Store:** Centralized feature management

### Phase 3: Production Scaling
* **Kubernetes:** Container orchestration for high availability
* **API Gateway:** Rate limiting, authentication, and request routing
* **Database Integration:** Persistent storage for predictions and feedback
* **Batch Processing:** Asynchronous prediction jobs for large datasets

### Phase 4: Enterprise Features
* **Security:** OAuth authentication and API key management
* **Compliance:** GDPR compliance for data handling
* **Multi-tenancy:** Isolated model deployments per customer
* **Auto-scaling:** Dynamic resource allocation based on load

---

## 15. Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation
- Ensure CI passes before merging

---

## 16. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 17. Acknowledgments

- Iris dataset from the UCI Machine Learning Repository
- Scikit-learn for machine learning algorithms
- FastAPI for API framework
- Weights & Biases for experiment tracking
- Conda for environment management
- Render for cloud deployment
