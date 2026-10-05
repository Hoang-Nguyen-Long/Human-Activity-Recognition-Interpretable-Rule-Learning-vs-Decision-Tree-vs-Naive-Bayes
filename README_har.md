# Human Activity Recognition: Interpretable Rule Learning vs Decision Tree vs Naive Bayes

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![SWI-Prolog](https://img.shields.io/badge/SWI--Prolog-ILP-E61B23)

This project classifies six everyday activities (walking, walking upstairs, walking downstairs, sitting, standing, laying) from smartphone accelerometer and gyroscope data, using the [UCI Human Activity Recognition dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

It compares three learners with very different ideas of what a "model" is:

- **Decision Tree**: propositional, axis-aligned splits over all 561 features.
- **Gaussian Naive Bayes**: probabilistic, assumes features are independent given the activity.
- **PyGoL-style ILP**: learns human-readable Prolog rules from symbolic background knowledge, run through SWI-Prolog.


---

## Results

All models are trained on the official UCI HAR training set (21 subjects, 7,352 windows) and evaluated on the official test set (9 different subjects, 2,947 windows). Because train and test subjects never overlap, the scores measure how well each model generalises to **new people**.

### Per-activity F1

| Activity | Decision Tree | Naive Bayes | ILP rules |
|---|:-:|:-:|:-:|
| LAYING | **1.000** | 0.74 | **1.000** |
| SITTING | **0.803** | 0.65 | 0.118 |
| STANDING | **0.838** | 0.83 | 0.102 |
| WALKING | **0.879** | 0.83 | 0.546 |
| WALKING_DOWNSTAIRS | **0.870** | 0.70 | 0.573 |
| WALKING_UPSTAIRS | 0.820 | **0.84** | 0.339 |
| **Macro F1** | **0.868** | 0.77 | 0.446 |

<!-- Naive Bayes values are read from the comparison chart. Re-run NaiveBayes_HAR.ipynb and replace them with exact figures. -->

### Overall

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | MCC |
|---|:-:|:-:|:-:|:-:|:-:|
| Decision Tree | **0.871** | **0.871** | **0.868** | **0.868** | **0.845** |
| Gaussian Naive Bayes | ≈0.77 | ≈0.79 | ≈0.77 | ≈0.77 | — |
| ILP rules (one-vs-rest) | — | 0.389 | 0.568 | 0.446 | 0.305 |

The ILP learner builds a separate yes/no rule for each activity, so its scores are averages of six binary problems. Its per-activity "accuracy" (mean 0.731) is inflated by the many easy negatives in each binary task, so it is left out of the table above and should not be compared with the six-class accuracy of the other two models. F1 is the fairer comparison.

### Rules learned

```prolog
laying(X)             :- tgravityacc_mean_x(X, low).
sitting(X)            :- anglexgravitymean(X, high).
standing(X)           :- tgravityacc_mean_y(X, high).
walking(X)            :- tbodyaccjerk_std_x(X, high).
walking_downstairs(X) :- fbodyaccjerk_bandsenergy_116(X, high).
walking_upstairs(X)   :- fbodyaccjerk_bandsenergy_116(X, high).
```

The rules show both the strength and the limits of the approach:

- **LAYING is solved by one literal.** When the phone lies flat, gravity no longer points along its X axis, and that single fact separates laying from everything else perfectly.
- **SITTING and STANDING fail.** Both rules have negative MCC on the test set, meaning they do worse than guessing. Each rule was learned to separate the target from a mixed pool of all other activities, mostly walking, so it never learned to tell sitting and standing apart.
- **Upstairs and downstairs get the same rule.** A single high-jerk literal detects "walking on stairs" but cannot say which direction.

### Key findings

1. **The Decision Tree wins clearly** (macro F1 0.868) because it can combine many of the 561 features in one model.
2. **Naive Bayes is held back by correlated features.** UCI HAR contains many near-duplicate features (for example, several gravity statistics that move together), which violates the independence assumption.
3. **SITTING vs STANDING is the hardest pair for every model.** Both are static, upright postures with similar gravity orientation and very low motion.
4. **Interpretability has a cost here.** Single-literal rules are fully readable and trivially checkable by a domain expert, but they cannot capture the multi-feature patterns needed for the harder activities.

---

## Method

**Data preparation** (`prepare_ucihar.py`)

- Loads the 561 pre-computed features (time-domain, frequency-domain and angle features from 2.56 s windows at 50 Hz), activity labels and subject IDs.
- Makes duplicate feature names unique and maps activity IDs to names.
- Writes `ucihar_train.csv`, `ucihar_test.csv` and `ucihar_combined.csv`, keeping the official subject-based split.

**Exploratory analysis** (`EDA_HAR.ipynb`)

- Class and subject distributions, feature-group overview.
- Random Forest importance and mutual information rankings, used to choose features for the ILP background knowledge.
- Ten engineered features (e.g. jerk magnitude, gyroscope magnitude, gravity/body ratio) with statistical significance tests.
- Correlation heatmaps, PCA, t-SNE and static-vs-dynamic separation.

**Decision Tree** (`DecisionTree_HAR.ipynb`): `max_depth=20`, `min_samples_leaf=5`. The final tree has depth 15, 127 leaves and uses 102 of the 561 features. Includes a depth sweep from 3 to 24.

**Naive Bayes** (`NaiveBayes_HAR.ipynb`): Gaussian NB with default settings, plus confidence analysis and one-vs-rest ROC curves.

**ILP rule learning** (`PyGoL_HAR.ipynb`)

- For each activity, three candidate features are chosen from the EDA rankings and domain knowledge.
- Each feature is discretised to `high` / `low` using a threshold from the training data only: the midpoint between the target activity's mean and the mean of all other activities.
- 200 positive and 200 negative training windows are asserted as Prolog facts via `pyswip`.
- Every single-literal rule is scored on the training examples, and the one with the best F1 is kept and evaluated on the test set.

**Comparison** (`Comparison_HAR.ipynb`) merges the three result files into summary tables, a bar chart, a radar chart and an F1 heatmap.

---

## Repository structure

```
.
├── prepare_ucihar.py         # Builds train/test/combined CSVs from the raw UCI files
├── EDA_HAR.ipynb             # Exploratory analysis and feature engineering
├── DecisionTree_HAR.ipynb    # Decision Tree model
├── NaiveBayes_HAR.ipynb      # Gaussian Naive Bayes model
├── PyGoL_HAR.ipynb           # ILP rule learning with SWI-Prolog
├── Comparison_HAR.ipynb      # Combined comparison of all three models
├── assets/                   # Figures used in this README
├── requirements.txt
└── README.md
```

Running the notebooks creates an `outputs/` folder with all plots and result CSVs. It is git-ignored.

---

## Getting started

### 1. Install SWI-Prolog

The ILP notebook calls Prolog through `pyswip`. Install [SWI-Prolog](https://www.swi-prolog.org/Download.html) (developed with 9.0.4) and make sure `swipl` is on your `PATH`.

### 2. Install Python dependencies

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download and prepare the data

Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). The download contains a second zip, `UCI HAR Dataset.zip`. Unzip that too, so a folder called `UCI HAR Dataset/` sits in the repository root. Then run:

```bash
python prepare_ucihar.py
```

### 4. Run the notebooks

```
EDA_HAR → DecisionTree_HAR → NaiveBayes_HAR → PyGoL_HAR → Comparison_HAR
```

The three model notebooks are independent. `Comparison_HAR.ipynb` reads their result files, so run it last.

---

## Limitations and future work

- The ILP learner searches only single-literal rules over three hand-picked features per activity. Allowing conjunctions (e.g. `standing(X) :- a(X, high), b(X, low)`) and adding negatives from the most confusable activity, such as SITTING for STANDING, are the obvious next steps.
- The ILP learner is trained on 400 examples per activity, while the Decision Tree and Naive Bayes use all 7,352.
- The ILP rules are evaluated one-vs-rest, so they never produce a single six-class prediction. Combining them into a rule list with a default class would allow a like-for-like accuracy comparison.
- The Decision Tree depth sweep uses the test set, so it is exploratory and was not used to choose the final model.

---

## References

- Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A public domain dataset for human activity recognition using smartphones. *ESANN 2013*.
- Muggleton, S. (1991). Inductive logic programming. *New Generation Computing*, 8, 295–318.

## Authors
