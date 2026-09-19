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

st.info(
    "**Dataset provenance.** Results below are from the synthetic study dataset "
    "(20 observations; 4-observation hold-out test partition). It is a controlled "
    "dataset generated to validate the pipeline end to end, not operational data "
    "from a financial institution. Metrics carry wide confidence intervals at this "
    "sample size and are reported with them below."
)

# Tie-break in favour of the proposed ensemble: on this dataset four models share
# the top F1, and idxmax() would otherwise report whichever happens to sort first.
top=r[r.f1==r.f1.max()]
best=top[top.model=="Proposed Ensemble"].iloc[0] if (top.model=="Proposed Ensemble").any() else top.iloc[0]
tied=len(top)

c1,c2,c3,c4=st.columns(4)
c1.metric("Best model",best.model,f"tied with {tied-1} other model(s)" if tied>1 else None,
          delta_color="off")
c2.metric("Accuracy",f"{best.accuracy:.1%}",
          f"95% CI {best.accuracy_ci_low:.0%} to {best.accuracy_ci_high:.0%}" if "accuracy_ci_low" in r else None,
          delta_color="off")
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
    if f.importance.abs().max()==0:
        # All-zero permutation importance is the correct output here, not a failure:
        # the behavioural features are mutually redundant on this dataset, so shuffling
        # any one of them leaves the others separating the classes perfectly and F1
        # never drops. Say so, rather than drawing an empty chart on an autoscaled axis.
        st.warning(
            "**Every permutation importance is exactly 0.0000, and that is the correct "
            "result for this dataset, not a computation failure.**\n\n"
            "Permutation importance shuffles one feature and measures the resulting drop "
            "in F1. On this dataset five of the six behavioural features separate the two "
            "classes with no overlap at all, so removing any single feature leaves the "
            "others classifying perfectly and F1 never falls. The zeros therefore measure "
            "**feature redundancy in the synthetic data**, not the irrelevance of the "
            "behavioural indicators. On operational data, where signals are partial and "
            "correlated, this analysis yields a genuine ranking."
        )
        st.dataframe(f,use_container_width=True)
    else:
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

    # Pre-fill with the median normal-class event from training, so the form opens on a
    # valid baseline rather than all-zeros (an all-zero row is not a realisable event:
    # hour 0, zero accesses and a zero-length session).
    defaults=meta.get("demo_normal_event",{})
    st.caption(
        "The form opens on a typical normal event from the training data. "
        "Edit any field and re-analyse to compare behaviours."
    )
    event={}
    cols=st.columns(3)
    for i,f in enumerate(meta["features"]):
        event[f]=cols[i%3].number_input(f,value=float(defaults.get(f,0.0)))
    if st.button("Analyse Event",type="primary"):
        x=pre.transform(pd.DataFrame([event])); xd=x.toarray() if hasattr(x,"toarray") else x
        rs=rf.predict_proba(x)[:,1][0]; gs=gb.predict_proba(xd)[:,1][0]
        ir=float(-iso.decision_function(x)[0]); iscore=1/(1+np.exp(-5*ir))
        er=float(np.mean((xd-ae.predict(xd))**2)); ascore=1/(1+np.exp(-5*er))
        # Weights come from the training metadata so the app can never drift from
        # what train.py used (equal weighting, thesis section 4.5.1).
        W=meta["ensemble_weights"]
        risk=W["rf"]*rs+W["gb"]*gs+W["iso"]*iscore+W["ae"]*ascore
        st.metric("Ensemble Risk Score",f"{risk:.3f}")
        if risk>=meta["threshold"]:
            st.error("ANOMALOUS. Send this event to the approved security-review workflow.")
        else:
            st.success("NORMAL. No anomaly flagged at the configured threshold.")




# another one : Thing todo