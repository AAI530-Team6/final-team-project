import argparse
import json
import numpy as np
import pandas as pd
import joblib


ACTIVITY_MAP = {
    0: "Other",
    1: "Lying",
    2: "Sitting",
    3: "Standing",
    4: "Walking",
    5: "Running",
    6: "Cycling",
    7: "Nordic walking",
    9: "Watching TV",
    10: "Computer work",
    11: "Car driving",
    12: "Ascending stairs",
    13: "Descending stairs",
    16: "Vacuum cleaning",
    17: "Ironing",
    18: "Folding laundry",
    19: "House cleaning",
    20: "Playing soccer",
    24: "Rope jumping",
}


def load_feature_cols(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        cols = json.load(f)
    if not isinstance(cols, list) or not all(isinstance(x, str) for x in cols):
        raise ValueError("feature_json must contain a JSON list of strings")
    return cols


def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    BASE = pd.Timestamp("2025-01-01 00:00:00")

    df = df.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)

    # elapsed seconds since the start of each subject recording
    df["t0"] = df.groupby("subject_id")["timestamp"].transform("min")
    df["elapsed_sec"] = df["timestamp"] - df["t0"]

    # artificial absolute datetime for Tableau
    df["datetime"] = BASE + pd.to_timedelta(df["elapsed_sec"], unit="s")
    df["date"] = df["datetime"].dt.date

    return df

def add_derived_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # magnitudes (acc16)
    df["hand_acc_mag"] = np.sqrt(df["hand_acc16_x"]**2 + df["hand_acc16_y"]**2 + df["hand_acc16_z"]**2)
    df["chest_acc_mag"] = np.sqrt(df["chest_acc16_x"]**2 + df["chest_acc16_y"]**2 + df["chest_acc16_z"]**2)
    df["ankle_acc_mag"] = np.sqrt(df["ankle_acc16_x"]**2 + df["ankle_acc16_y"]**2 + df["ankle_acc16_z"]**2)

    # hr normalization per subject
    df["hr_norm"] = df.groupby("subject_id")["heart_rate"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    return df

def drop_labels_if_needed(df: pd.DataFrame, drop_labels: str) -> pd.DataFrame:
    if drop_labels is None:
        return df
    drop_labels = drop_labels.strip()
    if not drop_labels:
        return df
    drop_ids = [int(x.strip()) for x in drop_labels.split(",") if x.strip()]
    return df[~df["activity_id_1"].isin(drop_ids)].copy()


def downsample_to_rate(df: pd.DataFrame, hz: int) -> pd.DataFrame:
    """
    Downsample by time bucketing using datetime.
    For hz=10, bucket to 100ms. For hz=20, bucket to 50ms, etc.
    """
    df = df.copy()
    ms = int(round(1000 / hz))
    if ms <= 0:
        raise ValueError("Invalid hz")
    df["dt_bucket"] = df["datetime"].dt.floor(f"{ms}ms")

    out = (
        df.sort_values(["subject_id", "datetime"])
          .drop_duplicates(subset=["subject_id", "dt_bucket"], keep="first")
          .reset_index(drop=True)
    )
    return out


def window_features(w: np.ndarray) -> np.ndarray:
    """
    Same as notebook: returns concatenated stats per channel.
    Input w shape: (win_size, num_raw_features)
    Output length: 12 * num_raw_features
    """
    mean = w.mean(axis=0)
    std  = w.std(axis=0)
    mn   = w.min(axis=0)
    mx   = w.max(axis=0)
    med  = np.median(w, axis=0)
    q75  = np.percentile(w, 75, axis=0)
    q25  = np.percentile(w, 25, axis=0)
    iqr  = q75 - q25

    energy = (w**2).mean(axis=0)

    jw = np.diff(w, axis=0)
    jerk_mean = jw.mean(axis=0)
    jerk_std  = jw.std(axis=0)
    jerk_energy = (jw**2).mean(axis=0)

    wd = w - mean
    fft = np.fft.rfft(wd, axis=0)
    psd = (np.abs(fft)**2)
    mid = max(1, psd.shape[0] // 2)
    low_band = psd[:mid].mean(axis=0)
    high_band = psd[mid:].mean(axis=0)

    return np.concatenate([
        mean, std, mn, mx, med, iqr,
        energy,
        jerk_mean, jerk_std, jerk_energy,
        low_band, high_band
    ], axis=0).astype(np.float32)


def predict_windows_and_project_to_1hz(
    df_ds: pd.DataFrame,
    model,
    raw_feature_cols: list[str],
    sample_rate_hz: int,
    win_seconds: int,
    step_seconds: int
) -> pd.DataFrame:
    """
    1) Build windows on downsampled timeline (e.g., 10 Hz)
    2) Predict per window using model trained on window_features
    3) Project predictions onto a 1 Hz timeline (one row per second per subject)
    """
    # Validate columns
    missing = [c for c in raw_feature_cols if c not in df_ds.columns]
    if missing:
        raise ValueError(f"Missing columns in data required by feature_json: {missing[:10]} ... total missing {len(missing)}")

    df_ds = df_ds.sort_values(["subject_id", "datetime"]).copy()

    win_size = int(win_seconds * sample_rate_hz)
    step = int(step_seconds * sample_rate_hz)
    if win_size <= 1 or step <= 0:
        raise ValueError("Invalid window or step size")

    # Prepare 1Hz scaffold
    df_ds["datetime_sec"] = df_ds["datetime"].dt.floor("s")
    df_1hz = (
        df_ds.sort_values(["subject_id", "datetime"])
             .drop_duplicates(subset=["subject_id", "datetime_sec"], keep="first")
             .reset_index(drop=True)
    )

    # fulfill these from window predictions
    pred_id = np.full(len(df_1hz), -1, dtype=np.int32)
    pred_prob = np.full(len(df_1hz), np.nan, dtype=np.float32)

    df_1hz["_row_idx"] = np.arange(len(df_1hz), dtype=np.int64)

    # For fast lookup from second to row index per subject
    for sid, g in df_ds.groupby("subject_id", sort=False):
        g = g.sort_values("datetime")
        arr = g[raw_feature_cols].to_numpy(dtype=np.float32)
        times = g["datetime"].to_numpy()

        if len(g) < win_size:
            continue

        # Build window feature matrix and remember window end times
        feats_list = []
        win_end_times = []

        for st in range(0, len(g) - win_size + 1, step):
            w = arr[st:st + win_size]
            feats_list.append(window_features(w))
            win_end_times.append(times[st + win_size - 1])

        Xw = np.vstack(feats_list)
        yhat = model.predict(Xw)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xw).max(axis=1).astype(np.float32)
        else:
            proba = np.full(len(yhat), np.nan, dtype=np.float32)

        win_end_times = pd.to_datetime(np.array(win_end_times)).floor("s")

        # Take one prediction per second: last window that ends in that second
        win_df = pd.DataFrame({
            "subject_id": sid,
            "datetime_sec": win_end_times,
            "activity_pred_id": yhat.astype(np.int32),
            "activity_pred_prob": proba
        })
        win_df = (
            win_df.sort_values(["datetime_sec"])
                 .groupby(["subject_id", "datetime_sec"], as_index=False)
                 .tail(1)
        )

        # Now align to df_1hz for this subject and forward fill
        h = df_1hz[df_1hz["subject_id"] == sid][["subject_id", "datetime_sec", "_row_idx"]].copy()
        h = h.sort_values("datetime_sec")

        merged = h.merge(win_df, on=["subject_id", "datetime_sec"], how="left")
        merged["activity_pred_id"] = merged["activity_pred_id"].ffill().bfill()
        merged["activity_pred_prob"] = merged["activity_pred_prob"].ffill().bfill()


        idx = merged["_row_idx"].to_numpy()
        pred_id[idx] = merged["activity_pred_id"].fillna(-1).to_numpy(dtype=np.int32)
        pred_prob[idx] = merged["activity_pred_prob"].to_numpy(dtype=np.float32)

    df_1hz.drop(columns=["_row_idx"], inplace=True)
    df_1hz["activity_pred_id"] = pred_id
    df_1hz["activity_pred_prob"] = pred_prob
    df_1hz["activity_pred_name"] = df_1hz["activity_pred_id"].map(ACTIVITY_MAP).fillna(df_1hz["activity_pred_id"].astype(str))
    df_1hz["activity_true_name"] = df_1hz["activity_id_1"].map(ACTIVITY_MAP).fillna(df_1hz["activity_id_1"].astype(str))

    # Tableau helper buckets
    df_1hz["minute_bucket"] = df_1hz["datetime"].dt.floor("min")
    df_1hz["bucket_30s"] = df_1hz["datetime"].dt.floor("30s")

    return df_1hz


def add_next_activity(df_1hz: pd.DataFrame) -> pd.DataFrame:
    df = df_1hz.copy()
    df = df.sort_values(["subject_id", "datetime"])

    df["next_datetime"] = df["datetime"] + pd.to_timedelta(30, unit="s")
    df["next_datetime_sec"] = df["next_datetime"].dt.floor("s")

    lookup = df[["subject_id", "datetime_sec", "activity_pred_id", "activity_pred_name", "activity_pred_prob"]].copy()
    lookup = lookup.rename(columns={
        "datetime_sec": "next_datetime_sec",
        "activity_pred_id": "next_activity_pred_id",
        "activity_pred_name": "next_activity_pred_name",
        "activity_pred_prob": "next_activity_pred_prob",
    })

    out = df.merge(lookup, on=["subject_id", "next_datetime_sec"], how="left")
    return out


def add_daily_and_benchmark(df_1hz: pd.DataFrame) -> pd.DataFrame:
    df = df_1hz.copy()

    # Minutes in activity per subject per day per predicted activity
    daily = (
        df.groupby(["subject_id", "date", "activity_pred_name"], as_index=False)
          .agg(seconds_in_activity=("datetime", "count"))
    )
    daily["minutes_in_activity"] = daily["seconds_in_activity"] / 60.0

    daily_total = (
        df.groupby(["subject_id", "date"], as_index=False)
          .agg(seconds_total=("datetime", "count"))
    )
    daily_total["patient_minutes_today"] = daily_total["seconds_total"] / 60.0

    bench = (
        daily.groupby(["date", "activity_pred_name"], as_index=False)
             .agg(benchmark_minutes_in_activity=("minutes_in_activity", "mean"))
    )

    bench_total = (
        daily_total.groupby(["date"], as_index=False)
                  .agg(benchmark_minutes_today=("patient_minutes_today", "mean"))
    )

    last_rows = (
        df.sort_values(["subject_id", "date", "datetime"])
          .groupby(["subject_id", "date"], as_index=False)
          .tail(1)[["subject_id", "date", "datetime", "activity_pred_name"]]
          .rename(columns={"datetime": "last_activity_datetime", "activity_pred_name": "last_activity_name"})
    )

    out = df.merge(
        daily[["subject_id", "date", "activity_pred_name", "minutes_in_activity"]],
        on=["subject_id", "date", "activity_pred_name"],
        how="left",
    )
    out = out.merge(
        daily_total[["subject_id", "date", "patient_minutes_today"]],
        on=["subject_id", "date"],
        how="left",
    )
    out = out.merge(
        bench,
        on=["date", "activity_pred_name"],
        how="left",
    )
    out = out.merge(
        bench_total,
        on=["date"],
        how="left",
    )
    out = out.merge(
        last_rows,
        on=["subject_id", "date"],
        how="left",
    )
    return out



def main():
    parser = argparse.ArgumentParser(description="Build Tableau-ready CSV from PAMAP2 using window-trained ExtraTrees.")
    parser.add_argument("--input_parquet", required=True, help="Path to pamap2_merged.parquet")
    parser.add_argument("--model_path", required=True, help="Path to saved ExtraTrees model joblib file")
    parser.add_argument("--feature_json", required=True, help="Path to feature_cols.json (raw cols used inside windows)")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--drop_labels", default="0,24", help="Comma-separated labels to drop, e.g. '0,24' or '' to keep all")
    parser.add_argument("--sample_rate_hz", type=int, default=10, help="Downsample rate used in training, default 10")
    parser.add_argument("--win_seconds", type=int, default=10, help="Window length in seconds, default 10")
    parser.add_argument("--step_seconds", type=int, default=5, help="Window step in seconds, default 5")

    args = parser.parse_args()

    df = pd.read_parquet(args.input_parquet)
    df = add_datetime(df)
    df = drop_labels_if_needed(df, args.drop_labels)

    df = add_derived_features_for_training(df)

    raw_feature_cols = load_feature_cols(args.feature_json)
    model = joblib.load(args.model_path)

    # Downsample to match training (e.g., 10 Hz)
    df_ds = downsample_to_rate(df, hz=args.sample_rate_hz)

    # Predict using windows, project to 1 Hz timeline for Tableau
    df_1hz = predict_windows_and_project_to_1hz(
        df_ds=df_ds,
        model=model,
        raw_feature_cols=raw_feature_cols,
        sample_rate_hz=args.sample_rate_hz,
        win_seconds=args.win_seconds,
        step_seconds=args.step_seconds
    )

    df_1hz = add_next_activity(df_1hz)
    df_out = add_daily_and_benchmark(df_1hz)

    df_out.to_csv(args.output_csv, index=False)
    print("Saved:", args.output_csv)
    print("Rows:", len(df_out), "Cols:", len(df_out.columns))


if __name__ == "__main__":
    main()