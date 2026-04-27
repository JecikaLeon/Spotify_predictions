import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


TRAIN_URL = "https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2026/main/datasets/dataTrain_Spotify.csv"
TEST_URL = "https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2026/main/datasets/dataTest_Spotify.csv"

TARGET = "popularity"
ID_COL = "Unnamed: 0"
N_SPLITS = 5
SEED = 42
MODEL_CONFIGS = [
    {
        "iterations": 4500,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 6,
        "random_seed_offset": 0,
    },
    {
        "iterations": 5200,
        "learning_rate": 0.025,
        "depth": 7,
        "l2_leaf_reg": 8,
        "random_seed_offset": 1000,
    },
]
FINAL_MODEL_CONFIGS = [
    {
        "iterations": 760,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 6,
        "random_seed": 2026,
    },
    {
        "iterations": 1200,
        "learning_rate": 0.025,
        "depth": 7,
        "l2_leaf_reg": 8,
        "random_seed": 3026,
    },
    {
        "iterations": 950,
        "learning_rate": 0.028,
        "depth": 8,
        "l2_leaf_reg": 9,
        "random_seed": 4026,
    },
]


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


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

    # Composite keys catch repeated songs that appear with different ids or genres.
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


def add_frequency_features(train_features, test_features):
    train_features = train_features.copy()
    test_features = test_features.copy()

    freq_cols = [
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

    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)
    for col in freq_cols:
        counts = combined[col].value_counts(dropna=False)
        train_features[f"{col}_freq"] = train_features[col].map(counts).astype(float)
        test_features[f"{col}_freq"] = test_features[col].map(counts).astype(float)

    return train_features, test_features


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


def add_target_stats(train_part, y_part, valid_part, test_part, alpha=8.0):
    train_part = train_part.copy()
    valid_part = valid_part.copy()
    test_part = test_part.copy()

    global_mean = float(np.mean(y_part))
    train_with_y = train_part.copy()
    train_with_y["_target"] = np.asarray(y_part)

    for keys in TARGET_STAT_KEYS:
        name = "te_" + "_".join(keys)
        stats = (
            train_with_y.groupby(keys, dropna=False)["_target"]
            .agg(["sum", "count", "std"])
            .reset_index()
        )

        merged_train = train_part.merge(stats, on=keys, how="left", sort=False)
        sums = merged_train["sum"].to_numpy(float)
        counts = merged_train["count"].to_numpy(float)
        leave_one_out_count = np.maximum(counts - 1, 0)

        train_part[f"{name}_mean"] = (
            (sums - np.asarray(y_part)) + global_mean * alpha
        ) / (leave_one_out_count + alpha)
        train_part[f"{name}_count"] = leave_one_out_count
        train_part[f"{name}_std"] = merged_train["std"].fillna(0).to_numpy(float)

        stats[f"{name}_mean"] = (stats["sum"] + global_mean * alpha) / (
            stats["count"] + alpha
        )
        stats[f"{name}_count"] = stats["count"]
        stats[f"{name}_std"] = stats["std"].fillna(0)
        stats = stats[keys + [f"{name}_mean", f"{name}_count", f"{name}_std"]]

        for frame in [valid_part, test_part]:
            encoded = frame.merge(stats, on=keys, how="left", sort=False)
            frame[f"{name}_mean"] = encoded[f"{name}_mean"].fillna(global_mean).to_numpy(float)
            frame[f"{name}_count"] = encoded[f"{name}_count"].fillna(0).to_numpy(float)
            frame[f"{name}_std"] = encoded[f"{name}_std"].fillna(0).to_numpy(float)

    return train_part, valid_part, test_part


def apply_known_song_overrides(predictions, train_features, y, test_features):
    predictions = predictions.copy()
    already_replaced = np.zeros(len(predictions), dtype=bool)
    override_rules = [
        (["track_id"], 1.0),
        (["artist_album_track"], 2.0),
        (["album_track"], 1.0),
    ]

    train_lookup = train_features.copy()
    train_lookup["_target"] = np.asarray(y)

    for keys, max_std in override_rules:
        stats = (
            train_lookup.groupby(keys, dropna=False)["_target"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        merged = test_features.merge(stats, on=keys, how="left", sort=False)
        use = merged["mean"].notna() & ~already_replaced
        if max_std is not None:
            use &= merged["std"].isna() | (merged["std"] <= max_std)

        replaced = int(use.sum())
        if replaced:
            predictions[use.to_numpy()] = merged.loc[use, "mean"].to_numpy(float)
            already_replaced |= use.to_numpy()
            print(f"Override {keys}: {replaced} rows")

    return predictions


def model_frame(df):
    return df.drop(columns=[col for col in MODEL_DROP_COLS if col in df.columns])


def train_full_data_models(X, y, X_test, cat_cols):
    X_full, _, X_te = add_target_stats(X, y.to_numpy(), X, X_test)
    X_full_model = model_frame(X_full)
    X_te_model = model_frame(X_te)
    cat_idx = [X_full_model.columns.get_loc(col) for col in cat_cols]

    train_pool = Pool(X_full_model, y, cat_features=cat_idx)
    test_pool = Pool(X_te_model, cat_features=cat_idx)

    predictions = np.zeros(len(X_test))
    for config_id, config in enumerate(FINAL_MODEL_CONFIGS, 1):
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=config["iterations"],
            learning_rate=config["learning_rate"],
            depth=config["depth"],
            l2_leaf_reg=config["l2_leaf_reg"],
            random_seed=config["random_seed"],
            allow_writing_files=False,
            verbose=200,
        )
        model.fit(train_pool)
        pred = np.clip(model.predict(test_pool), 0, 100)
        predictions += pred / len(FINAL_MODEL_CONFIGS)
        print(f"Full-data model {config_id} done")

    return predictions


def main():
    n_splits = N_SPLITS
    verbose = 200

    train = pd.read_csv(TRAIN_URL)
    test = pd.read_csv(TEST_URL, index_col=0)

    y = train[TARGET].astype(float)

    X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
    X_test = test.copy()

    X = add_features(X)
    X_test = add_features(X_test)
    X, X_test = add_frequency_features(X, X_test)

    cat_cols = model_frame(X).select_dtypes(include=["object"]).columns.tolist()
    print("Categorical columns:", cat_cols)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va, X_te = add_target_stats(
            X.iloc[tr_idx],
            y.iloc[tr_idx].to_numpy(),
            X.iloc[va_idx],
            X_test,
        )
        X_tr_model = model_frame(X_tr)
        X_va_model = model_frame(X_va)
        X_te_model = model_frame(X_te)

        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        cat_idx = [X_tr_model.columns.get_loc(col) for col in cat_cols]

        train_pool = Pool(X_tr_model, y_tr, cat_features=cat_idx)
        valid_pool = Pool(X_va_model, y_va, cat_features=cat_idx)
        test_pool = Pool(X_te_model, cat_features=cat_idx)

        fold_valid_pred = np.zeros(len(X_va_model))
        fold_test_pred = np.zeros(len(X_te_model))

        for config_id, config in enumerate(MODEL_CONFIGS, 1):
            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=config["iterations"],
                learning_rate=config["learning_rate"],
                depth=config["depth"],
                l2_leaf_reg=config["l2_leaf_reg"],
                random_seed=SEED + fold + config["random_seed_offset"],
                od_type="Iter",
                od_wait=250,
                allow_writing_files=False,
                verbose=verbose,
            )

            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

            pred_va = np.clip(model.predict(valid_pool), 0, 100)
            fold_valid_pred += pred_va / len(MODEL_CONFIGS)
            fold_test_pred += np.clip(model.predict(test_pool), 0, 100) / len(MODEL_CONFIGS)

            rmse = root_mean_squared_error(y_va, pred_va)
            print(f"Fold {fold} model {config_id} RMSE: {rmse:.5f}")

        oof[va_idx] = fold_valid_pred
        fold_rmse = root_mean_squared_error(y_va, fold_valid_pred)
        print(f"Fold {fold} ensemble RMSE: {fold_rmse:.5f}")

        test_pred += fold_test_pred / n_splits

    overall_rmse = root_mean_squared_error(y, np.clip(oof, 0, 100))
    print(f"\nOOF RMSE before overrides: {overall_rmse:.5f}")
    pd.DataFrame({"oof": np.clip(oof, 0, 100), "target": y}).to_csv(
        "oof_predictions_V5.csv",
        index=False,
    )

    model_only_submission = pd.DataFrame(
        {
            "ID": test.index,
            "Popularity": np.clip(test_pred, 0, 100),
        }
    )
    model_only_submission.to_csv("test_submission_file_V5_model_only.csv", index=False)

    final_pred = apply_known_song_overrides(test_pred, X, y, X_test)
    final_pred = np.clip(final_pred, 0, 100)

    submission = pd.DataFrame({"ID": test.index, "Popularity": final_pred})

    output_path = "test_submission_file_V5.csv"
    submission.to_csv(output_path, index=False)
    print(submission.head())
    print("Saved: test_submission_file_V5_model_only.csv")
    print(f"Saved: {output_path}")

    full_train_pred = train_full_data_models(X, y, X_test, cat_cols)
    full_train_submission = pd.DataFrame(
        {"ID": test.index, "Popularity": np.clip(full_train_pred, 0, 100)}
    )
    full_train_submission.to_csv("test_submission_file_V6_full_train.csv", index=False)

    blended_pred = 0.45 * np.clip(test_pred, 0, 100) + 0.55 * np.clip(full_train_pred, 0, 100)
    blended_submission = pd.DataFrame({"ID": test.index, "Popularity": blended_pred})
    blended_submission.to_csv("test_submission_file_V6_blend_model_only.csv", index=False)

    blended_override_pred = apply_known_song_overrides(blended_pred, X, y, X_test)
    blended_override_submission = pd.DataFrame(
        {"ID": test.index, "Popularity": np.clip(blended_override_pred, 0, 100)}
    )
    blended_override_submission.to_csv("test_submission_file_V6_blend.csv", index=False)
    print("Saved: test_submission_file_V6_full_train.csv")
    print("Saved: test_submission_file_V6_blend_model_only.csv")
    print("Saved: test_submission_file_V6_blend.csv")


if __name__ == "__main__":
    main()
