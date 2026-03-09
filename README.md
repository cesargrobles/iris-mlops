# Iris Species Classification — Automated Flower Identification

**Course:** MLOps: Master in Business Analytics and Data Science  
**Status:** Production-Ready (Modularized Pipeline)

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

## 4. Academic Purpose & ML Approach

This repository is a teaching scaffold for **Machine Learning Operations (MLOps)**. We transition from a fragile Jupyter Notebook into a testable software engineering architecture.

* **Separation of Concerns:** Every step (Loading, Cleaning, Validating, Training) has a dedicated, single-purpose Python module.
* **Fail-Fast Security Gates:** `validate.py` blocks missing values and schema violations before expensive compute begins.
* **Leakage Prevention:** Data is split *before* fitting any feature transformations or the model.
* **Deployable Artifacts:** The orchestrator bundles preprocessing and the classifier into a single `.joblib` file, preventing training-serving skew.

### Future Roadmap (Upcoming Sessions)
* Move `SETTINGS` into `config.yaml` with environment-based overrides.
* Replace `print()` statements with structured logging via the `logging` module.
* Add MLflow for experiment tracking and model registry.
* Containerise and serve predictions via a FastAPI application.

---

## 5. Repository Structure

```text
.
├── README.md
├── config.yaml
├── environment.yml
├── pytest.ini
├── data
│   ├── raw/
│   │   └── iris.csv
│   └── processed/
│       └── clean.csv
├── models/
│   └── model.joblib
├── notebooks/
│   ├── 00_iris_analysis_vLegacy.ipynb
│   └── 01_iris_analysis_vExp.ipynb
├── reports/
│   └── predictions.csv
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── clean_data.py
│   ├── validate.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── utils.py
│   └── main.py
└── tests/
    ├── test_load_data.py
    ├── test_clean_data.py
    ├── test_validate.py
    ├── test_features.py
    ├── test_train.py
    ├── test_evaluate.py
    ├── test_infer.py
    └── test_main.py
```

---

## 6. How to Run & Test

### Step 1: Environment Setup
Build and activate the Conda environment:
```
conda env create -f environment.yml
conda activate mlops_kickoff
```

### Step 2: Exploratory Sandbox (The Lab Bench)
Before running the automated pipeline, interactively explore the data and debug your custom modules using the provided Jupyter Notebook.

* **The Sandbox (`notebooks/01_iris_analysis_vExp.ipynb`):** The data scientist's "lab bench" for interactive inspection, rapid iteration, and viewing intermediate states. Runs entirely in memory and **does not** write production artifacts to disk.
* **The Orchestrator (`src/main.py`):** The automated "factory". The *only* entry point authorised to write canonical production artifacts to disk.

Launch the sandbox:
```
jupyter notebook notebooks/01_iris_analysis_vExp.ipynb
```

### Step 3: Run the Test Suite
Ensure the codebase is sound and all pipeline contracts are unbroken:
```
python -m pytest -q
```
> (You should see 100% passing tests!)

### Step 4: Execute the Orchestrator
Run the full end-to-end pipeline — cleans data, trains the model, and generates artifacts:
```
python -m src.main
```
> **Note:** You may see a `FutureWarning` from scikit-learn related to `KBinsDiscretizer`. This is not an error — it is a deprecation notice about a default parameter that will change in sklearn 1.9. The pipeline runs correctly and all outputs are generated as expected.

---

## 7. Outputs Generated

1. `data/processed/clean.csv` — the deterministically cleaned input data
2. `models/model.joblib` — the deployable pipeline artifact (preprocessor + classifier)
3. `reports/predictions.csv` — the inference log containing predictions and class probabilities
