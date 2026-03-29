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

MODEL_PATH = "svd_model.pkl"
K_CANDIDATES = [10, 20, 30, 50, 100]


def main():
    # 1. Load data
    print("Loading data...")
    ratings_df, movies_df = load_data()

    # 2. Split rating rows into train / val / test
    train_df, val_df, test_df = split_data(ratings_df)
    print(
        f"Split sizes — "
        f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    # 3. Tune k using only train and val — test set is never touched here
    best_k, _ = tune_k(train_df, val_df, K_CANDIDATES)

    # 4. Retrain on train + val combined using the best k
    print(f"\nRetraining on train+val with best k={best_k}...")
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    R_final, user_ids, movie_ids = build_matrix(train_val_df)
    reconstructed_final, Vt = train_svd(R_final, best_k)

    # 5. Evaluate once on the held-out test set
    test_rmse, test_count = compute_rmse(
        reconstructed_final, user_ids, movie_ids, test_df
    )
    print(f"\nFinal test RMSE = {test_rmse:.4f}  (n={test_count})")

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

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nSaved model to {MODEL_PATH}  (k={best_k}, test_rmse={test_rmse:.4f})")


if __name__ == "__main__":
    main()
