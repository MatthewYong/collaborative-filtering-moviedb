import pickle
import numpy as np


MODEL_PATH = "svd_model.pkl"


def ask_user_for_ratings(movies_to_ask):
    collected = {}

    print("Rate these movies from 1 to 5.")
    print("Type 0 if you don't know the movie.\n")

    # Collect the User ratings on different movies
    for movie_id, title in movies_to_ask:
        while True:
            raw = input(f"{title}: ")
            try:
                rating = int(raw)
                if rating in [0, 1, 2, 3, 4, 5]:
                    break
                print("Please enter 0, 1, 2, 3, 4, or 5.")
            except ValueError:
                print("Please enter a number.")

        if rating != 0:
            collected[movie_id] = rating

        if len(collected) >= 5:
            break

    return collected


def main():
    # 1. Load and process the Saved pickle model
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    Vt = artifact["Vt"]
    movie_ids = artifact["movie_ids"]
    movies_df = artifact["movies_df"]
    movie_popularity = artifact["movie_popularity"]

    movie_id_to_title = dict(zip(movies_df["movie_id"], movies_df["movie_title"]))
    movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    # 2. Prepare a list of suggested movies
    popular_movie_ids = [
        mid for mid, _ in sorted(
            movie_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        if mid in movie_id_to_index and mid in movie_id_to_title
    ]

    movies_to_ask = [
        (mid, movie_id_to_title[mid])
        for mid in popular_movie_ids[:100]
    ]

    user_ratings = ask_user_for_ratings(movies_to_ask)

    if len(user_ratings) < 5:
        print("\nNot enough ratings collected.")
        return

    rated_indices = [movie_id_to_index[mid] for mid in user_ratings.keys()]
    ratings_values = np.array(list(user_ratings.values()), dtype=float)

    # 3. Find the latent vector for movies the user rated
    # The equation  = movie_factors × user_vector ≈ ratings
    movie_factors = Vt[:, rated_indices].T
    user_vector, *_ = np.linalg.lstsq(movie_factors, ratings_values, rcond=None)

    # 4. Predict preference scores for all movies
    preds = user_vector @ Vt

    # 5. Clean up already rated movies
    for mid in user_ratings.keys():
        preds[movie_id_to_index[mid]] = -np.inf

    top_indices = np.argsort(preds)[::-1][:10]

    # 6. Show top 10 Recommended movies
    print("\nTop 10 recommended movies:\n")
    for rank, idx in enumerate(top_indices, start=1):
        movie_id = movie_ids[idx]
        title = movie_id_to_title.get(movie_id, f"movie_id={movie_id}")
        score = preds[idx]
        print(f"{rank}. {title}  (predicted score: {score:.3f})")


if __name__ == "__main__":
    main()