import argparse
import shutil
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor, Pool

from model_runtime import build_training_matrix, save_artifact


TRAIN_URL = "https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2026/main/datasets/dataTrain_Spotify.csv"
TEST_URL = "https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2026/main/datasets/dataTest_Spotify.csv"

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


def read_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path, index_col=0) if test_path else None
    return train, test


def train_models(X_model, y, cat_cols):
    cat_idx = [X_model.columns.get_loc(col) for col in cat_cols]
    train_pool = Pool(X_model, y, cat_features=cat_idx)

    models = []
    for model_id, config in enumerate(FINAL_MODEL_CONFIGS, 1):
        print(f"Training final model {model_id}/{len(FINAL_MODEL_CONFIGS)}")
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
        models.append(model)
    return models


def copy_runtime(output_paths):
    source = Path(__file__).with_name("model_runtime.py")
    for output_path in output_paths:
        target = Path(output_path).parent / "model_runtime.py"
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(
        description="Train and export the Spotify popularity model as uncompressed pickle."
    )
    parser.add_argument("--train", default=TRAIN_URL, help="Training CSV path or URL.")
    parser.add_argument("--test", default=TEST_URL, help="Optional test CSV path or URL.")
    parser.add_argument(
        "--flask-output",
        default="deployment/flask/model.pkl",
        help="Output model path for Flask deployment.",
    )
    parser.add_argument(
        "--lambda-output",
        default="deployment/lambda/model.pkl",
        help="Output model path for AWS Lambda deployment.",
    )
    args = parser.parse_args()

    train, test = read_data(args.train, args.test)
    X_model, y, metadata = build_training_matrix(train, test)
    models = train_models(X_model, y, metadata["cat_cols"])

    artifact = {
        "model_version": "spotify-catboost-v1",
        "models": models,
        **metadata,
    }

    output_paths = [args.flask_output, args.lambda_output]
    for output_path in output_paths:
        save_artifact(artifact, output_path)
        print(f"Saved uncompressed pickle: {output_path}")

    copy_runtime(output_paths)
    print("Copied model_runtime.py to both deployment folders.")


if __name__ == "__main__":
    main()
