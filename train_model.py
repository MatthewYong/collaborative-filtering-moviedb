"""
Offline training script for the SVD movie recommender.

Usage:
    python train_model.py

Stages
------
1. Load ratings + movies data
2. Split ratings rows  →  ~75% train / ~5% validation / ~20% test
3. Tune k on (train, val)  –  test set is never touched here
4. Retrain on train + val combined using the best k
5. Evaluate once on the held-out test set
6. Save artifact to svd_model.pkl
"""

import pickle

import pandas as pd

from src.train_svd import (
    load_data,
    split_data,
    build_matrix,
    train_svd,
    compute_rmse,
    tune_k,
)

DEFAULT_MODEL_PATH = "svd_model.pkl"
DEFAULT_K_CANDIDATES = [10, 20, 30, 50, 100]
DEFAULT_TEST_SPLIT = 0.20
DEFAULT_VAL_SPLIT = 0.05
DEFAULT_RANDOM_STATE = 42


def run_training(
    model_path=DEFAULT_MODEL_PATH,
    k_candidates=None,
    test_split=DEFAULT_TEST_SPLIT,
    val_split=DEFAULT_VAL_SPLIT,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Run the full offline training pipeline.

    Parameters
    ----------
    model_path    : path where the artifact will be saved
    k_candidates  : list of k values to try during tuning
    test_split    : fraction of ratings held out for final test evaluation
    val_split     : fraction of ratings used for validation during k tuning
    random_state  : random seed (currently used by split_data via train_test_split)

    Returns
    -------
    dict with keys:
        best_k, val_rmse, test_rmse, n_users, n_movies, n_ratings,
        tuning_results, model_path, split_sizes
    """
    if k_candidates is None:
        k_candidates = DEFAULT_K_CANDIDATES

    # 1. Load data
    ratings_df, movies_df = load_data()

    # 2. Split
    train_df, val_df, test_df = split_data(
        ratings_df, test_split=test_split, val_split=val_split, random_state=random_state
    )

    # 3. Tune k
    best_k, tuning_results = tune_k(train_df, val_df, k_candidates)
    best_val_rmse = min(rmse for _, rmse, _ in tuning_results)

    # 4. Retrain on train + val combined
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    R_final, user_ids, movie_ids = build_matrix(train_val_df)
    reconstructed_final, Vt = train_svd(R_final, best_k)

    # 5. Evaluate on the held-out test set (exactly once)
    test_rmse, test_count = compute_rmse(
        reconstructed_final, user_ids, movie_ids, test_df
    )

    # 6. Build movie_popularity from the full dataset for serving
    movie_popularity = (
        ratings_df.groupby("movie_id")["rating"]
        .count()
        .sort_values(ascending=False)
    )

    artifact = {
        "R": R_final,
        "Vt": Vt,
        "movie_ids": movie_ids,
        "user_ids": user_ids,
        "movies_df": movies_df,
        "movie_popularity": movie_popularity.to_dict(),
        "k": best_k,
        "test_rmse": test_rmse,
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    return {
        "best_k": best_k,
        "val_rmse": round(best_val_rmse, 4),
        "test_rmse": round(test_rmse, 4),
        "n_users": len(user_ids),
        "n_movies": len(movie_ids),
        "n_ratings": len(ratings_df),
        "tuning_results": [
            {"k": k, "val_rmse": round(rmse, 4), "n": count}
            for k, rmse, count in tuning_results
        ],
        "model_path": model_path,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }


def main():
    print("Loading data...")
    result = run_training()
    print(
        f"\nSaved model to {result['model_path']}  "
        f"(k={result['best_k']}, test_rmse={result['test_rmse']:.4f})"
    )


if __name__ == "__main__":
    main()
