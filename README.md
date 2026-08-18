# UBA Security Analytics — Complete Python Deployment

## Included
- End-to-end preprocessing and feature engineering
- Rule-based baseline
- Isolation Forest
- Autoencoder reconstruction detector
- Random Forest
- Gradient Boosted Decision Tree
- Proposed ensemble
- SMOTE on training data
- 10-fold cross-validation
- Accuracy, precision, recall, F1, FPR and ROC-AUC
- Confusion matrices
- Feature importance
- Model persistence
- Batch CSV scoring
- Single-event scoring
- Streamlit dashboard

## Run
pip install -r requirements.txt
python train.py --data data/events.csv --target label --dataset CERT
streamlit run dashboard.py

## Data
The training CSV should contain behavioural variables and a binary `label`
(0 = normal, 1 = anomalous). Replace `data/events.csv` with the approved
research dataset when available.
