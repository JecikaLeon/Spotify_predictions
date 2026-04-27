import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool


TARGET = "popularity"
ID_COL = "Unnamed: 0"

FREQ_COLS = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "track_genre",
    "artist_track",
    "album_track",
    "artist_album_track",
    "duration_artist_track",
    "genre_artist",
]

TARGET_STAT_KEYS = [
    ["track_id"],
    ["artist_album_track"],
    ["album_track"],
    ["artists"],
    ["album_name"],
    ["track_name"],
    ["track_genre"],
    ["genre_artist"],
]

MODEL_DROP_COLS = [
    "track_id",
    "artist_track",
    "album_track",
    "artist_album_track",
    "duration_artist_track",
    "genre_artist",
]

OVERRIDE_RULES = [
    (["track_id"], 1.0),
    (["artist_album_track"], 2.0),
    (["album_track"], 1.0),
]


def add_features(df):
    df = df.copy()

    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].fillna("missing").astype(str)

    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].astype(str)

    for col in ["key", "mode", "time_signature"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["duration_min"] = df["duration_ms"] / 60000.0
    df["duration_log"] = np.log1p(df["duration_ms"])
    df["tempo_per_min"] = df["tempo"] / df["duration_min"].clip(lower=0.1)

    df["dance_x_energy"] = df["danceability"] * df["energy"]
    df["valence_x_energy"] = df["valence"] * df["energy"]
    df["acoustic_x_instrumental"] = df["acousticness"] * df["instrumentalness"]
    df["energy_x_loudness"] = df["energy"] * df["loudness"]

    for col in ["artists", "album_name", "track_name"]:
        text = df[col].astype(str)
        words = text.str.split().str.len().clip(lower=1)
        df[f"{col}_len"] = text.str.len()
        df[f"{col}_words"] = words
        df[f"{col}_chars_per_word"] = df[f"{col}_len"] / words

    df["n_artists"] = df["artists"].str.count(r";|,|&| and ") + 1

    df["artist_track"] = df["artists"] + " || " + df["track_name"]
    df["album_track"] = df["album_name"] + " || " + df["track_name"]
    df["artist_album_track"] = (
        df["artists"] + " || " + df["album_name"] + " || " + df["track_name"]
    )
    df["duration_artist_track"] = (
        df["duration_ms"].astype(str) + " || " + df["artists"] + " || " + df["track_name"]
    )
    df["genre_artist"] = df["track_genre"] + " || " + df["artists"]

    return df


def model_frame(df):
    return df.drop(columns=[col for col in MODEL_DROP_COLS if col in df.columns])


def fit_frequency_maps(train_features, reference_features=None):
    if reference_features is None:
        combined = train_features
    else:
        combined = pd.concat([train_features, reference_features], axis=0, ignore_index=True)

    return {
        col: combined[col].value_counts(dropna=False).astype(float).to_dict()
        for col in FREQ_COLS
    }


def apply_frequency_maps(features, freq_maps):
    features = features.copy()
    for col, counts in freq_maps.items():
        features[f"{col}_freq"] = features[col].map(counts).fillna(0).astype(float)
    return features


def fit_target_stats(train_part, y_part, alpha=8.0):
    global_mean = float(np.mean(y_part))
    train_with_y = train_part.copy()
    train_with_y["_target"] = np.asarray(y_part)

    target_stats = {}
    for keys in TARGET_STAT_KEYS:
        name = "te_" + "_".join(keys)
        stats = (
            train_with_y.groupby(keys, dropna=False)["_target"]
            .agg(["sum", "count", "std"])
            .reset_index()
        )
        stats[f"{name}_mean"] = (stats["sum"] + global_mean * alpha) / (
            stats["count"] + alpha
        )
        stats[f"{name}_count"] = stats["count"]
        stats[f"{name}_std"] = stats["std"].fillna(0)
        target_stats[name] = {
            "keys": keys,
            "frame": stats[keys + [f"{name}_mean", f"{name}_count", f"{name}_std"]],
        }

    return target_stats, global_mean


def apply_target_stats(features, target_stats, global_mean):
    features = features.copy()
    for name, payload in target_stats.items():
        keys = payload["keys"]
        stats = payload["frame"]
        encoded = features.merge(stats, on=keys, how="left", sort=False)
        features[f"{name}_mean"] = encoded[f"{name}_mean"].fillna(global_mean).to_numpy(float)
        features[f"{name}_count"] = encoded[f"{name}_count"].fillna(0).to_numpy(float)
        features[f"{name}_std"] = encoded[f"{name}_std"].fillna(0).to_numpy(float)
    return features


def apply_target_stats_leave_one_out(train_part, y_part, target_stats, global_mean, alpha=8.0):
    train_part = train_part.copy()
    y_values = np.asarray(y_part)

    for name, payload in target_stats.items():
        keys = payload["keys"]
        stats = payload["frame"].copy()
        mean_col = f"{name}_mean"
        count_col = f"{name}_count"
        std_col = f"{name}_std"

        stats["_sum_smooth"] = stats[mean_col] * (stats[count_col] + alpha)
        stats["_sum"] = stats["_sum_smooth"] - global_mean * alpha

        merged = train_part.merge(stats, on=keys, how="left", sort=False)
        sums = merged["_sum"].to_numpy(float)
        counts = merged[count_col].to_numpy(float)
        leave_one_out_count = np.maximum(counts - 1, 0)

        train_part[mean_col] = ((sums - y_values) + global_mean * alpha) / (
            leave_one_out_count + alpha
        )
        train_part[count_col] = leave_one_out_count
        train_part[std_col] = merged[std_col].fillna(0).to_numpy(float)

    return train_part


def fit_override_stats(train_features, y):
    train_lookup = train_features.copy()
    train_lookup["_target"] = np.asarray(y)
    override_stats = {}
    for keys, max_std in OVERRIDE_RULES:
        name = "||".join(keys)
        frame = (
            train_lookup.groupby(keys, dropna=False)["_target"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        override_stats[name] = {"keys": keys, "max_std": max_std, "frame": frame}
    return override_stats


def apply_known_song_overrides(predictions, features, override_stats):
    predictions = predictions.copy()
    already_replaced = np.zeros(len(predictions), dtype=bool)

    for payload in override_stats.values():
        keys = payload["keys"]
        max_std = payload["max_std"]
        stats = payload["frame"]
        merged = features.merge(stats, on=keys, how="left", sort=False)
        use = merged["mean"].notna() & ~already_replaced
        if max_std is not None:
            use &= merged["std"].isna() | (merged["std"] <= max_std)

        if int(use.sum()):
            predictions[use.to_numpy()] = merged.loc[use, "mean"].to_numpy(float)
            already_replaced |= use.to_numpy()

    return predictions


def build_training_matrix(train_raw, test_raw=None):
    y = train_raw[TARGET].astype(float)
    X = train_raw.drop(columns=[TARGET, ID_COL], errors="ignore")
    X = add_features(X)

    X_reference = None
    if test_raw is not None:
        X_reference = add_features(test_raw.copy())

    freq_maps = fit_frequency_maps(X, X_reference)
    X = apply_frequency_maps(X, freq_maps)

    target_stats, global_mean = fit_target_stats(X, y.to_numpy())
    X_encoded = apply_target_stats_leave_one_out(X, y.to_numpy(), target_stats, global_mean)
    X_model = model_frame(X_encoded)
    cat_cols = X_model.select_dtypes(include=["object"]).columns.tolist()

    override_stats = fit_override_stats(X, y)

    metadata = {
        "feature_columns": X_model.columns.tolist(),
        "cat_cols": cat_cols,
        "freq_maps": freq_maps,
        "target_stats": target_stats,
        "global_mean": global_mean,
        "override_stats": override_stats,
    }
    return X_model, y, metadata


def prepare_inference_matrix(records, artifact):
    if isinstance(records, dict):
        records = [records]

    X = pd.DataFrame(records)
    X = X.drop(columns=[TARGET, ID_COL], errors="ignore")
    X = add_features(X)
    X = apply_frequency_maps(X, artifact["freq_maps"])
    X_for_overrides = X.copy()
    X = apply_target_stats(X, artifact["target_stats"], artifact["global_mean"])
    X = model_frame(X)
    X = X.reindex(columns=artifact["feature_columns"], fill_value=0)
    return X, X_for_overrides


def predict_records(records, artifact, apply_overrides=True):
    X, X_for_overrides = prepare_inference_matrix(records, artifact)
    cat_idx = [X.columns.get_loc(col) for col in artifact["cat_cols"]]
    pool = Pool(X, cat_features=cat_idx)

    predictions = np.zeros(len(X))
    for model in artifact["models"]:
        predictions += np.clip(model.predict(pool), 0, 100) / len(artifact["models"])

    if apply_overrides:
        predictions = apply_known_song_overrides(
            predictions,
            X_for_overrides,
            artifact["override_stats"],
        )

    return np.clip(predictions, 0, 100).tolist()


def save_artifact(artifact, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_artifact(model_path):
    with Path(model_path).open("rb") as file:
        return pickle.load(file)
