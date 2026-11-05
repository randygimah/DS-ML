# Data Science & Machine Learning in Python

Notebooks and exercises from a Python for Data Science and Machine Learning course. They cover the core workflow end to end: exploring and cleaning data, feature engineering, training and evaluating supervised and unsupervised models, and serving a trained model through an API.

**Stack:** Python · pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn · Plotly · PyCaret · FastAPI

## What's covered

| Topic | Notebook(s) | Techniques |
|---|---|---|
| Python, NumPy & pandas fundamentals | `exercises/original/unit2.ipynb` | Slicing, array operations, DataFrames, descriptive statistics |
| Exploratory data analysis & visualisation | `exercises/original/mod2ex.ipynb`, `guides/Guide - Exploring Data with Pandas and Matplot.ipynb` | Distributions, correlations and plotting with Matplotlib and Seaborn |
| Statistics | `exercises/original/day2.ipynb` | Two-sample t-test with `scipy.stats` |
| Data preparation & feature engineering | `exercises/original/mod4ex.ipynb` | Cleaning, train/test split, one-hot encoding, scaling |
| Linear regression | `exercises/original/day3mod5.ipynb`, `exercises/original/Linear Regression - Singapore Real Estate .ipynb` | `LinearRegression`, feature scaling, MAE / MSE / R² |
| Classification | `exercises/original/day3mod6.ipynb`, `guides/Guide - Classification modelling -  Iris Species.ipynb` | Logistic regression, decision trees, random forests, k-NN; confusion matrices, classification reports, ROC curves |
| Clustering | `exercises/original/day4mod8.ipynb`, `guides/Guide-Clustering-KMeans.ipynb`, `guides/PyCaret - Clustering Demo.ipynb` | k-means, choosing *k* with silhouette scores, PyCaret clustering |
| Dimensionality reduction | `guides/PCA.ipynb`, `guides/Foundational_Dimensionality reduction demo.ipynb` | PCA from scratch and with scikit-learn, explained variance, kernel PCA, t-SNE, PCA + logistic regression pipeline tuned with `GridSearchCV` |
| Model deployment | `QADSMLP-learner/10_Deploying Models & Insights/` | Random forest price model saved with `joblib` and served through a FastAPI `/predict` endpoint |

Reference guides and demonstrations are in `guides/`. Previous exercise
solutions are preserved in `exercises/original/`; new attempts belong in
[exercises/refresher/](exercises/refresher/README.md). The Singapore regression
notebook identifies itself as a solution and is archived with the old solutions.
Notebook contents, outputs and existing local edits were preserved during the move.

## Repository layout

```text
guides/                 Reference notebooks and demonstrations
exercises/original/     Previous course attempts and solutions
exercises/refresher/    Fresh attempts and a progress checklist
notes/                  Explanations and learning notes
outputs/original/       Existing generated figures/documents
QADSMLP-learner/         Existing deployment example and local data location
```

Exercise-question PDFs are currently outside this repository in
`/Users/randygyimah/Documents/QADSMLP-learner/exercise_pdfs/`.
Keep those prompts open while working in a fresh notebook; consult the old
solutions only after attempting the exercise.

## Datasets

The CSV files aren't committed (see `.gitignore`). Some notebooks use scikit-learn's built-in datasets (Iris, Wine, Digits). The others expect their CSVs in `QADSMLP-learner/data/`:

`housing_data.csv`, `loan_data.csv`, `renfe_trains.csv`, `Real estate.csv`, `mortgage_applicants.csv`, `train_viz.csv`, `titanic.csv`, `iris.csv`, `experimental_investment_performance.csv`

## Notebook paths after organisation

Old notebooks have been moved without rewriting their code. Their relative
paths assume the repository root as the working directory. Before running an
archived notebook from its new folder, run this setup in a temporary cell:

```python
from pathlib import Path
import os

repo_root = next(
    p for p in (Path.cwd(), *Path.cwd().parents)
    if (p / ".git").exists() and (p / "QADSMLP-learner").is_dir()
)
os.chdir(repo_root)
```

Most CSVs are expected under `QADSMLP-learner/data/`; the pandas guide instead
uses `data/titanic.csv`. These datasets were not moved or downloaded as part of
organisation. On this machine, course CSVs are available separately under
`/Users/randygyimah/Documents/QADSMLP-learner/data/`; supply missing files or
set an explicit data directory in a fresh attempt. Do not assume all historical
notebooks run unchanged: they can contain old errors or missing datasets.

For new notebooks, prefer explicit `DATA_DIR` and `OUTPUT_DIR` paths. Save new
plots under `outputs/refresher/` rather than overwriting archived outputs.

## Running locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install jupyter pandas numpy scipy scikit-learn matplotlib seaborn plotly
jupyter notebook
```

To try the model API:

```bash
pip install fastapi uvicorn joblib
cd "QADSMLP-learner/10_Deploying Models & Insights"
python retrain_model.py         # trains and saves the model
uvicorn main:app --reload
# then open http://127.0.0.1:8000/predict/1?ds=BARCELONA&vc=Turista&fr=Promo
```

PyCaret is optional and only needed for the PyCaret demo. Some reference notebooks were written for older scikit-learn versions and may need small import changes.
