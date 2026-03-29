import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from src.paths import RATINGS_PATH, MOVIES_PATH

ITEM_COLUMNS = [
    "movie_id", "movie_title", "release_date", "video_release_date",
    "imdb_url", "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
    "War", "Western",
]


# ---------------------------------------------------------------------------
# Step 1 – Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load ratings and movies DataFrames from disk."""
    ratings_df = pd.read_csv(
        RATINGS_PATH,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )[["user_id", "movie_id", "rating"]]

    movies_df = pd.read_csv(
        MOVIES_PATH,
        sep="|",
        names=ITEM_COLUMNS,
        encoding="latin-1",
    )[["movie_id", "movie_title"]]

    return ratings_df, movies_df


# ---------------------------------------------------------------------------
# Step 2 – Data splitting  (rows of ratings_df, never the pivot matrix)
# ---------------------------------------------------------------------------

def split_data(ratings_df, test_split=0.20, val_split=0.05, random_state=42):
    """
    Split rating rows into train / validation / test sets.

    Parameters
    ----------
    test_split   : fraction of all ratings reserved for the test set (default 0.20)
    val_split    : fraction of all ratings reserved for validation (default 0.05)
    random_state : random seed for reproducible splits (default 42)

    Sizes (with defaults):
        train      : ~75 % of all ratings  (used to fit SVD during tuning)
        validation :  ~5 % of all ratings  (used to pick the best k)
        test       : ~20 % of all ratings  (evaluated exactly once at the end)

    The test set is held out completely and never touched during tuning.
    """
    train_val_df, test_df = train_test_split(
        ratings_df, test_size=test_split, random_state=random_state
    )
    # Convert absolute val_split fraction to a fraction of train_val
    val_size_of_trainval = val_split / (1.0 - test_split)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_of_trainval, random_state=random_state
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Step 3 – Build pivot matrix
# ---------------------------------------------------------------------------

def build_matrix(ratings_subset):
    """
    Build a user × movie pivot matrix from a ratings DataFrame subset.

    Returns
    -------
    R         : 2-D numpy array  (users × movies), zeros for missing ratings
    user_ids  : list of user IDs  (row labels)
    movie_ids : list of movie IDs (column labels)
    """
    R_df = ratings_subset.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        fill_value=0.0,
    )
    return R_df.values, R_df.index.tolist(), R_df.columns.tolist()


# ---------------------------------------------------------------------------
# Step 4 – Train SVD
# ---------------------------------------------------------------------------

def train_svd(R, k):
    """
    Fit TruncatedSVD with k latent factors on matrix R.

    Returns
    -------
    reconstructed : low-rank approximation of R  (U · Σ · Vt)
    Vt            : item factor matrix  (k × movies)
    """
    svd = TruncatedSVD(n_components=k, random_state=42)
    U_sigma = svd.fit_transform(R)   # shape: (users, k)
    Vt = svd.components_             # shape: (k, movies)
    reconstructed = U_sigma @ Vt     # shape: (users, movies)
    return reconstructed, Vt


# ---------------------------------------------------------------------------
# Step 5 – RMSE evaluation
# ---------------------------------------------------------------------------

def compute_rmse(reconstructed, user_ids, movie_ids, eval_df):
    """
    Compute RMSE between the reconstructed matrix and actual ratings.

    Assumption: entries whose user or movie does not appear in the training
    pivot matrix (cold-start) are skipped, because the reconstructed matrix
    has no row / column for them.

    Returns
    -------
    rmse  : float RMSE over all matched pairs
    count : number of pairs that were evaluated
    """
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    movie_index = {mid: i for i, mid in enumerate(movie_ids)}

    squared_errors = []
    for _, row in eval_df.iterrows():
        uid = row["user_id"]
        mid = row["movie_id"]
        actual = row["rating"]

        # Skip cold-start entries not seen during training
        if uid not in user_index or mid not in movie_index:
            continue

        predicted = reconstructed[user_index[uid], movie_index[mid]]
        squared_errors.append((predicted - actual) ** 2)

    if not squared_errors:
        return None, 0

    return float(np.sqrt(np.mean(squared_errors))), len(squared_errors)


# ---------------------------------------------------------------------------
# Step 6 – Hyperparameter tuning loop
# ---------------------------------------------------------------------------

def tune_k(train_df, val_df, k_candidates):
    """
    Try each candidate value of k. Train on train_df, score on val_df.

    Returns
    -------
    best_k        : k with the lowest validation RMSE
    tuning_results: list of (k, rmse, count) for every candidate
    """
    print("\n--- Hyperparameter Tuning (k = n_components) ---")

    # Build the training matrix once — all candidates share it
    R_train, user_ids, movie_ids = build_matrix(train_df)

    tuning_results = []
    for k in k_candidates:
        reconstructed, _ = train_svd(R_train, k)
        rmse, count = compute_rmse(reconstructed, user_ids, movie_ids, val_df)
        print(f"  k={k:>4}  val_RMSE={rmse:.4f}  (n={count})")
        tuning_results.append((k, rmse, count))

    best_k, best_rmse, _ = min(tuning_results, key=lambda x: x[1])
    print(f"\n  Best k = {best_k}  (val_RMSE = {best_rmse:.4f})")
    return best_k, tuning_results