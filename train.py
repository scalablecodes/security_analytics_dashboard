import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

SEED=42
BASE=Path(__file__).resolve().parent
MODEL=BASE/"models"; MODEL.mkdir(exist_ok=True)

def load_dataset(path,target):
    df=pd.read_csv(path)
    if target not in df.columns: raise ValueError(f"Missing target: {target}")
    df=df.dropna(subset=[target]).copy()
    y=df.pop(target)
    if not np.issubdtype(y.dtype,np.number):
        mp={"normal":0,"benign":0,"legitimate":0,"anomalous":1,"anomaly":1,"malicious":1,"attack":1}
        y=y.astype(str).str.lower().map(mp)
    y=pd.to_numeric(y,errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0,1}):
        raise ValueError("Target must contain binary 0/1 labels.")
    return df,y.astype(int)

def build_preprocessor(X):
    num=X.select_dtypes(include=np.number).columns.tolist()
    cat=[c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
        ("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                                 ("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
    ])

def calc(y,p,s):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {
      "accuracy":accuracy_score(y,p),"precision":precision_score(y,p,zero_division=0),
      "recall":recall_score(y,p,zero_division=0),"f1":f1_score(y,p,zero_division=0),
      "fpr":fp/(fp+tn) if fp+tn else 0,
      "roc_auc":roc_auc_score(y,s) if len(np.unique(y))==2 else np.nan,
      "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--target",default="label")
    ap.add_argument("--dataset",default="CERT"); args=ap.parse_args()
    X,y=load_dataset(args.data,args.target)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,stratify=y,random_state=SEED)

    pre=build_preprocessor(X)
    A=pre.fit_transform(Xtr); B=pre.transform(Xte)
    A_bal,y_bal=SMOTE(random_state=SEED).fit_resample(A,ytr)
    Ad=A_bal.toarray() if hasattr(A_bal,"toarray") else A_bal
    Bd=B.toarray() if hasattr(B,"toarray") else B

    rf=RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=SEED,n_jobs=-1)
    rf.fit(A_bal,y_bal); rf_s=rf.predict_proba(B)[:,1]
    gb=GradientBoostingClassifier(n_estimators=250,learning_rate=.05,max_depth=3,random_state=SEED)
    gb.fit(Ad,y_bal); gb_s=gb.predict_proba(Bd)[:,1]

    iso=IsolationForest(n_estimators=400,contamination="auto",random_state=SEED)
    iso.fit(A); raw=-iso.decision_function(B)
    iso_s=(raw-raw.min())/(raw.max()-raw.min()+1e-9)

    ae=MLPRegressor(hidden_layer_sizes=(64,32,16,32,64),activation="relu",
                    max_iter=500,early_stopping=True,random_state=SEED)
    ae.fit(Ad,Ad); err=np.mean((Bd-ae.predict(Bd))**2,axis=1)
    ae_s=(err-err.min())/(err.max()-err.min()+1e-9)

    # Rule baseline: fixed anomaly score threshold.
    rule_s=.5*iso_s+.5*ae_s
    # Proposed ensemble.
    ens_s=.40*rf_s+.35*gb_s+.15*iso_s+.10*ae_s

    models=[
      ("Rule-based baseline",rule_s,(rule_s>=.80).astype(int)),
      ("Isolation Forest",iso_s,(iso_s>=.80).astype(int)),
      ("Autoencoder",ae_s,(ae_s>=.80).astype(int)),
      ("Random Forest",rf_s,(rf_s>=.50).astype(int)),
      ("Gradient Boosted Decision Tree",gb_s,(gb_s>=.50).astype(int)),
      ("Proposed Ensemble",ens_s,(ens_s>=.50).astype(int))
    ]
    rows=[]
    for name,s,p in models: rows.append({"dataset":args.dataset,"model":name,**calc(yte,p,s)})
    pd.DataFrame(rows).to_csv(MODEL/f"{args.dataset}_performance.csv",index=False)

    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    cvrows=[]
    for name,m,Xcv in [("Random Forest",rf,A_bal),("Gradient Boosted Decision Tree",gb,Ad)]:
        vals=cross_val_score(m,Xcv,y_bal,cv=cv,scoring="accuracy")
        cvrows.append({"dataset":args.dataset,"model":name,"mean_accuracy":vals.mean(),"sd":vals.std()})
    pd.DataFrame(cvrows).to_csv(MODEL/f"{args.dataset}_cv.csv",index=False)

    # Permutation importance of RF on the test set.
    perm=permutation_importance(rf,B,yte,n_repeats=10,random_state=SEED,scoring="f1")
    names=pre.get_feature_names_out()
    fi=pd.DataFrame({"feature":names,"importance":perm.importances_mean}).sort_values("importance",ascending=False)
    fi.to_csv(MODEL/f"{args.dataset}_feature_importance.csv",index=False)

    joblib.dump(pre,MODEL/f"{args.dataset}_preprocessor.joblib")
    for n,m in {"rf":rf,"gb":gb,"iso":iso,"ae":ae}.items(): joblib.dump(m,MODEL/f"{args.dataset}_{n}.joblib")
    (MODEL/f"{args.dataset}_metadata.json").write_text(json.dumps({
        "dataset":args.dataset,"target":args.target,"features":X.columns.tolist(),
        "ensemble_weights":{"rf":.40,"gb":.35,"iso":.15,"ae":.10},
        "threshold":.50
    },indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print("Saved model artifacts to",MODEL)

if __name__=="__main__": main()
