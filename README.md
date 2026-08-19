# UBA Security Analytics — Complete Python Deployment

## Included
- End-to-end preprocessing and feature engineering
- Rule-based baseline
- Isolation Forest
- Autoencoder reconstruction detector
- Random Forest
- Gradient Boosted Decision Tree
- Proposed ensemble
- SMOTE on the training partition only
- 5-fold stratified cross-validation (see note below)
- Accuracy, precision, recall, F1, FPR and ROC-AUC
- Exact (Clopper-Pearson) 95% confidence intervals for accuracy and recall
- Confusion matrices
- Permutation feature importance
- Model persistence
- Single-event scoring via the dashboard
- Streamlit dashboard

## Note on cross-validation
Cross-validation is 5-fold, not 10-fold. With 20 observations (10 per class),
10 stratified folds leave too few observations per class per fold to be
meaningful. See `Chapter_Four_Run_Report.txt`.

## Note on sample size
`data/events.csv` is a synthetic 20-observation dataset built to validate the
pipeline end to end. The hold-out test partition is 4 observations, so the
reported metrics carry very wide confidence intervals — 100% accuracy on n=4
has an exact 95% interval of roughly 40%-100%. The CI columns in
`models/*_performance.csv` report this directly. These results demonstrate that
the pipeline runs correctly; they are not evidence of operational detection
performance.

## Run
pip install -r requirements.txt
python train.py --data data/events.csv --target label --dataset STUDY
streamlit run dashboard.py

## Data
The training CSV should contain behavioural variables and a binary `label`
(0 = normal, 1 = anomalous). Replace `data/events.csv` with the approved
research dataset when available.
