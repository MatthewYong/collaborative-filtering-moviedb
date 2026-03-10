import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# -----------------------------
# 1) Tiny "ratings table" (like u.data but very small)
# -----------------------------
ratings = pd.DataFrame(
    [
        # user, movie, rating
        ("U1", "M1", 5),
        ("U1", "M2", 3),
        ("U2", "M1", 4),
        ("U2", "M3", 2),
        ("U3", "M2", 5),
        ("U3", "M3", 4),
        # Note: some user-movie pairs are missing (unknown)
    ],
    columns=["user_id", "movie_id", "rating"],
)

print("\n=== Ratings table (input) ===")
print(ratings)

# -----------------------------
# 2) Build the user-item matrix R
#    Rows = users, Columns = movies
#    Missing ratings become 0 (unknown)
# -----------------------------
R_df = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating",
    fill_value=0.0,
)

# Ensure consistent column order
R_df = R_df[sorted(R_df.columns)]
R = R_df.to_numpy(dtype=np.float32)

print("\n=== User-Item Matrix R (what we factorize) ===")
print(R_df)



# -----------------------------
# 3) Factorize R using TruncatedSVD
#    R ≈ U @ Vt
# -----------------------------



k = 2  # number of latent factors (keep small so you can see it)
svd = TruncatedSVD(n_components=k, random_state=42)



# fit_transform does TWO things:
# - fit(): learn Vt (the components) from R
# - transform(): compute U for each user



U = svd.fit_transform(R)      # shape: (n_users, k)
Vt = svd.components_          # shape: (k, n_movies)

U_df = pd.DataFrame(U, index=R_df.index, columns=[f"factor_{i+1}" for i in range(k)])
Vt_df = pd.DataFrame(Vt, index=[f"factor_{i+1}" for i in range(k)], columns=R_df.columns)

print("\n=== U (user latent vectors) ===")
print(U_df.round(3))

print("\n=== Vt (movie latent vectors; components_) ===")
print(Vt_df.round(3))

# -----------------------------
# 4) Reconstruct predicted matrix R_hat
# -----------------------------
R_hat = U @ Vt
R_hat_df = pd.DataFrame(R_hat, index=R_df.index, columns=R_df.columns)

print("\n=== Reconstructed R_hat = U @ Vt (predicted scores) ===")
print(R_hat_df.round(3))

# -----------------------------
# 5) Example: recommend unseen movies for a user
# -----------------------------
user = "U1"
seen = set(ratings.loc[ratings["user_id"] == user, "movie_id"])
all_movies = list(R_df.columns)

candidates = [m for m in all_movies if m not in seen]
scores = R_hat_df.loc[user, candidates].sort_values(ascending=False)

print(f"\n=== Recommendations for {user} (unseen movies only) ===")
print(scores.round(3))
