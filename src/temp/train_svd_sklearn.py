from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml-100k"

RATINGS_PATH = DATA_DIR / "u.data"
MOVIES_PATH = DATA_DIR / "u.item"


def load_data():
    ratings = pd.read_csv(
        RATINGS_PATH,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )[["user_id", "movie_id", "rating"]]

    movies = pd.read_csv(
        MOVIES_PATH,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["movie_id", "title"],
    )

    return ratings, movies


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    ratings, movies = load_data()

    # Map ids to 0..n-1 indices for matrix building
    user_ids = ratings["user_id"].unique()
    movie_ids = ratings["movie_id"].unique()

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    ratings["user_idx"] = ratings["user_id"].map(user_id_to_idx)
    ratings["movie_idx"] = ratings["movie_id"].map(movie_id_to_idx)

    n_users = len(user_ids)
    n_movies = len(movie_ids)

    # Train/test split on ratings rows (simple + good enough for learning)
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

    # Build sparse-ish matrix as dense (943x1682 is fine)
    R = np.zeros((n_users, n_movies), dtype=np.float32)
    for row in train_df.itertuples(index=False):
        R[row.user_idx, row.movie_idx] = row.rating

    # Matrix factorization via TruncatedSVD
    # n_components = number of latent factors
    svd = TruncatedSVD(n_components=50, random_state=42)
    U = svd.fit_transform(R)          # users in latent space
    Vt = svd.components_              # movies in latent space

    # Reconstruct predicted ratings matrix
    R_hat = np.dot(U, Vt)

    # Evaluate on held-out ratings
    preds = []
    trues = []
    for row in test_df.itertuples(index=False):
        pred = R_hat[row.user_idx, row.movie_idx]
        # Clip predictions to rating scale
        pred = float(np.clip(pred, 1, 5))
        preds.append(pred)
        trues.append(row.rating)

    print(f"Users: {n_users}, Movies: {n_movies}")
    print(f"Train ratings: {len(train_df)}, Test ratings: {len(test_df)}")
    print("RMSE:", rmse(trues, preds))

    # Show a few predictions
    print("\nSample predictions:")
    for i in range(10):
        row = test_df.iloc[i]
        mid = row["movie_id"]
        title = movies.loc[movies["movie_id"] == mid, "title"].values[0]
        print(f"user={row['user_id']}, movie={title}, true={row['rating']}, pred={preds[i]:.3f}")


if __name__ == "__main__":
    main()
