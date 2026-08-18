import json
from pathlib import Path
import joblib,numpy as np,pandas as pd,streamlit as st,plotly.express as px

BASE=Path(__file__).resolve().parent; MODEL=BASE/"models"
st.set_page_config(page_title="UBA Security Analytics Dashboard",layout="wide")
st.title("User Behaviour Analytics (UBA) Security Dashboard")
st.caption("Behavioural anomaly detection, model comparison and live risk scoring")

perf=list(MODEL.glob("*_performance.csv"))
if not perf:
    st.warning("No trained model results found. Run train.py first.")
    st.stop()

datasets=sorted({p.stem.replace("_performance","") for p in perf})
dataset=st.sidebar.selectbox("Dataset",datasets)
r=pd.read_csv(MODEL/f"{dataset}_performance.csv")

best=r.loc[r.f1.idxmax()]
c1,c2,c3,c4=st.columns(4)
c1.metric("Best model",best.model); c2.metric("Accuracy",f"{best.accuracy:.1%}")
c3.metric("F1-score",f"{best.f1:.1%}"); c4.metric("ROC-AUC",f"{best.roc_auc:.3f}")

st.subheader("Model Comparison")
metric=st.selectbox("Performance metric",["accuracy","precision","recall","f1","fpr","roc_auc"])
fig=px.bar(r,x="model",y=metric,text=r[metric].round(3),title=f"{metric.upper()} by model")
st.plotly_chart(fig,use_container_width=True)

st.subheader("Confusion Matrix Components")
st.dataframe(r[["model","tn","fp","fn","tp"]],use_container_width=True)

st.subheader("Detailed Performance")
st.dataframe(r,use_container_width=True)

fi=MODEL/f"{dataset}_feature_importance.csv"
if fi.exists():
    st.subheader("Behavioural Feature Importance")
    f=pd.read_csv(fi).head(15)
    st.plotly_chart(px.bar(f.sort_values("importance"),x="importance",y="feature",orientation="h"),use_container_width=True)

st.divider()
st.header("Live Behavioural Event Scoring")
live_files=[
    MODEL/f"{dataset}_metadata.json",
    MODEL/f"{dataset}_preprocessor.joblib",
    MODEL/f"{dataset}_rf.joblib",
    MODEL/f"{dataset}_gb.joblib",
    MODEL/f"{dataset}_iso.joblib",
    MODEL/f"{dataset}_ae.joblib",
]
missing=[p.name for p in live_files if not p.exists()]
if missing:
    st.info(
        f"Live scoring is not available for the **{dataset}** dataset because the "
        f"following model artifacts are missing: `{', '.join(missing)}`. "
        f"Run `python train.py --data data/events.csv --target label --dataset {dataset}` "
        f"to generate the full model bundle, or switch to a dataset that has pre-trained "
        f"scoring artifacts (e.g. STUDY)."
    )
else:
    meta=json.loads(live_files[0].read_text())
    pre=joblib.load(live_files[1])
    rf=joblib.load(live_files[2])
    gb=joblib.load(live_files[3])
    iso=joblib.load(live_files[4])
    ae=joblib.load(live_files[5])

    event={}
    cols=st.columns(3)
    for i,f in enumerate(meta["features"]):
        event[f]=cols[i%3].number_input(f,value=0.0)
    if st.button("Analyse Event",type="primary"):
        x=pre.transform(pd.DataFrame([event])); xd=x.toarray() if hasattr(x,"toarray") else x
        rs=rf.predict_proba(x)[:,1][0]; gs=gb.predict_proba(xd)[:,1][0]
        ir=float(-iso.decision_function(x)[0]); iscore=1/(1+np.exp(-5*ir))
        er=float(np.mean((xd-ae.predict(xd))**2)); ascore=1/(1+np.exp(-5*er))
        risk=.40*rs+.35*gs+.15*iscore+.10*ascore
        st.metric("Ensemble Risk Score",f"{risk:.3f}")
        if risk>=meta["threshold"]:
            st.error("ANOMALOUS — send event to the approved security-review workflow.")
        else:
            st.success("NORMAL — no anomaly flagged at the configured threshold.")




# another one : Thing todo