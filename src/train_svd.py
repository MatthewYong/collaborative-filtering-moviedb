import pickle
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from paths import RATINGS_PATH, MOVIES_PATH

MODEL_PATH = "svd_model.pkl"


def main():
    # 1. Import and read data
    ratings_df = pd.read_csv(
        RATINGS_PATH,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )[["user_id", "movie_id", "rating"]]

    item_columns = [
        "movie_id",
        "movie_title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    movies_df = pd.read_csv(
        MOVIES_PATH,
        sep="|",
        names=item_columns,
        encoding="latin-1",
    )[["movie_id", "movie_title"]]

    ratings_with_titles = ratings_df.merge(movies_df, on="movie_id", how="left")

    # 2. Convert data into matrix
    R_df = ratings_with_titles.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        fill_value=0.0,
    )

    R = R_df.values

    # Parameter to set model accuracy vs speed
    k = 20
    svd = TruncatedSVD(n_components=k, random_state=42)

    svd.fit_transform(R)
    Vt = svd.components_

    movie_popularity = (
        ratings_df.groupby("movie_id")["rating"]
        .count()
        .sort_values(ascending=False)
    )

    artifact = {
        "R": R,
        "Vt": Vt,
        "movie_ids": R_df.columns.tolist(),
        "user_ids": R_df.index.tolist(),
        "movies_df": movies_df,
        "movie_popularity": movie_popularity.to_dict(),
        "k": k,
    }

    # 3. Save model for later to be used
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()