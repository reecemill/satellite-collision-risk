"""Build the data file for the interactive results explorer.

Repeats the steps in notebooks/03_model.ipynb: the same labels, the same
seed-42 train/test split, the trend features, and a cutoff chosen on
cross-validated training predictions only. Then it writes every test event's
updates, the baseline's and the model's verdicts, and the five-split results
to one JSON file.

    python scripts/build_explorer_data.py ../portfolio-site/collision/data.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

ROOT = Path(__file__).resolve().parent.parent
FORECAST_DAYS = 2
DANGER = -6


def load():
    df = pd.read_csv(ROOT / "Data" / "train_data.csv")
    earliest_update = df.groupby("event_id")["time_to_tca"].max()
    final_update = df.loc[df.groupby("event_id")["time_to_tca"].idxmin()].set_index("event_id")
    y = (final_update["risk"] >= DANGER)[earliest_update >= FORECAST_DAYS]

    observed = df[df["time_to_tca"] >= FORECAST_DAYS]
    latest = observed.loc[observed.groupby("event_id")["time_to_tca"].idxmin()].set_index("event_id")

    X = latest.loc[y.index].drop(columns=["mission_id"])
    X["c_object_type"] = X["c_object_type"].astype("category")

    obs = observed.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    obs["risk_change"] = obs.groupby("event_id")["risk"].diff()
    obs["miss_distance_change"] = obs.groupby("event_id")["miss_distance"].diff()
    last_rows = obs.groupby("event_id").tail(1).set_index("event_id")
    by_event = obs.groupby("event_id")
    trend = pd.DataFrame({
        "risk_change": last_rows["risk_change"],
        "miss_distance_change": last_rows["miss_distance_change"],
        "risk_max": by_event["risk"].max(),
        "risk_min": by_event["risk"].min(),
        "n_updates": by_event.size(),
    })
    can_change = final_update.loc[y.index, "time_to_tca"] < FORECAST_DAYS
    return df, y, latest, X, X.join(trend), trend, can_change


def choose_cutoff(model, X, y, seed):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probs = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
    cutoffs = np.linspace(0.01, 0.99, 99)
    return float(cutoffs[np.argmax([fbeta_score(y, probs >= c, beta=2) for c in cutoffs])])


def new_model():
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=42)


def five_splits(features, y, latest, can_change):
    rows = []
    for seed in range(5):
        tr, te = train_test_split(y.index, test_size=0.2, stratify=y, random_state=seed)
        model = new_model()
        cutoff = choose_cutoff(model, features.loc[tr], y.loc[tr], seed)
        model.fit(features.loc[tr], y.loc[tr])
        model_pred = pd.Series(model.predict_proba(features.loc[te])[:, 1] >= cutoff, index=te)
        base_pred = latest.loc[te, "risk"] >= DANGER
        y_te, hard = y.loc[te], can_change.loc[te]
        rows.append({
            "model_all": fbeta_score(y_te, model_pred, beta=2),
            "baseline_all": fbeta_score(y_te, base_pred, beta=2),
            "model_can_change": fbeta_score(y_te[hard], model_pred[hard], beta=2),
            "baseline_can_change": fbeta_score(y_te[hard], base_pred[hard], beta=2),
            "model_misses": int((~model_pred[hard] & y_te[hard]).sum()),
            "baseline_misses": int((~base_pred[hard] & y_te[hard]).sum()),
            "model_false_alarms": int((model_pred[hard] & ~y_te[hard]).sum()),
            "baseline_false_alarms": int((base_pred[hard] & ~y_te[hard]).sum()),
        })
    return {k: round(float(v), 3) for k, v in pd.DataFrame(rows).mean().items()}


def group_counts(y, latest, can_change):
    base = latest.loc[y.index, "risk"] >= DANGER
    out = {}
    for key, group in (("no_more_updates", ~can_change), ("can_change", can_change)):
        yy, pp = y[group], base[group]
        out[key] = {
            "events": int(group.sum()), "dangerous": int(yy.sum()),
            "caught": int((pp & yy).sum()), "missed": int((~pp & yy).sum()),
            "false_alarms": int((pp & ~yy).sum()), "f2": round(float(fbeta_score(yy, pp, beta=2)), 3),
        }
    return out


def main(out_path):
    df, y, latest, X, X2, trend, can_change = load()

    train_ids, test_ids = train_test_split(y.index, test_size=0.2, stratify=y, random_state=42)
    model = new_model()
    cutoff = choose_cutoff(model, X2.loc[train_ids], y.loc[train_ids], 42)
    model.fit(X2.loc[train_ids], y.loc[train_ids])
    probs = pd.Series(model.predict_proba(X2.loc[test_ids])[:, 1], index=test_ids)

    updates = df[df["event_id"].isin(test_ids)].sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    events = []
    for event_id, rows in updates.groupby("event_id"):
        t = trend.loc[event_id]
        events.append({
            "id": int(event_id),
            "dangerous": bool(y[event_id]),
            "baseline": bool(latest.loc[event_id, "risk"] >= DANGER),
            "model": bool(probs[event_id] >= cutoff),
            "prob": round(float(probs[event_id]), 4),
            "can_change": bool(can_change[event_id]),
            "object": str(latest.loc[event_id, "c_object_type"]),
            "risk_change": None if pd.isna(t["risk_change"]) else round(float(t["risk_change"]), 2),
            "risk_max": round(float(t["risk_max"]), 2),
            "n_updates": int(t["n_updates"]),
            # Each update: [days before closest approach, risk (log10), miss distance in meters]
            "updates": [[round(float(a), 3), round(float(r), 2), None if pd.isna(m) else int(round(m))]
                        for a, r, m in rows[["time_to_tca", "risk", "miss_distance"]].itertuples(index=False)],
        })

    data = {
        "forecast_days": FORECAST_DAYS,
        "danger": DANGER,
        "cutoff": round(cutoff, 2),
        "events_total": int(len(y)),
        "dangerous_total": int(y.sum()),
        "groups": group_counts(y, latest, can_change),
        "five_splits": {"latest_only": five_splits(X, y, latest, can_change),
                        "with_trends": five_splits(X2, y, latest, can_change)},
        "events": events,
    }
    Path(out_path).write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {len(events)} test events to {out_path} (cutoff {cutoff:.2f})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
