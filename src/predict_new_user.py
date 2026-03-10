import pickle
import random
import numpy as np
from sklearn.metrics import mean_squared_error


MODEL_PATH = "svd_model.pkl"


def ask_user_for_ratings(movies_to_ask, target_count=10):
    collected = {}

    print("Rate these movies from 1 to 5.")
    print("Type 0 if you don't know the movie.\n")

    # Collect the user ratings on different movies
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

        if len(collected) >= target_count:
            break

    return collected


def build_movies_to_ask(movie_popularity, movie_id_to_index, movie_id_to_title):
    # 2a. Keep only movies that exist in the trained matrix and have titles
    valid_movie_ids = [
        mid for mid in movie_popularity.keys()
        if mid in movie_id_to_index and mid in movie_id_to_title
    ]

    # 2b. Sort by popularity so we still prefer well-known movies
    sorted_valid_movie_ids = sorted(
        valid_movie_ids,
        key=lambda mid: movie_popularity[mid],
        reverse=True,
    )

    # 2c. Take a larger pool from the top popular movies
    # so the choices are still recognizable, but not always identical
    candidate_pool = sorted_valid_movie_ids[:300]

    # 2d. Shuffle that pool to randomize the order
    random.shuffle(candidate_pool)

    # 2e. Convert movie ids to (movie_id, movie_title)
    movies_to_ask = [
        (mid, movie_id_to_title[mid])
        for mid in candidate_pool
    ]

    return movies_to_ask


def main():
    # 1. Load and process the saved pickle model
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    Vt = artifact["Vt"]
    movie_ids = artifact["movie_ids"]
    movies_df = artifact["movies_df"]
    movie_popularity = artifact["movie_popularity"]

    movie_id_to_title = dict(zip(movies_df["movie_id"], movies_df["movie_title"]))
    movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    # 2. Prepare a more randomized list of suggested movies
    movies_to_ask = build_movies_to_ask(
        movie_popularity=movie_popularity,
        movie_id_to_index=movie_id_to_index,
        movie_id_to_title=movie_id_to_title,
    )

    # 3. Ask the user for 10 known ratings
    user_ratings = ask_user_for_ratings(movies_to_ask, target_count=10)

    if len(user_ratings) < 10:
        print("\nNot enough ratings collected.")
        return

    # 4. Split the 10 ratings into:
    #    - first 5 for building the latent user vector
    #    - remaining 5 for testing prediction quality
    rated_items = list(user_ratings.items())

    train_ratings = dict(rated_items[:5])
    test_ratings = dict(rated_items[5:10])

    # 5. Convert the first 5 training ratings into matrix indices and values
    train_indices = [movie_id_to_index[mid] for mid in train_ratings.keys()]
    train_values = np.array(list(train_ratings.values()), dtype=float)

    # 6. Find the latent vector for the new user
    #    Equation: movie_factors @ user_vector ≈ train_values
    movie_factors = Vt[:, train_indices].T
    user_vector, *_ = np.linalg.lstsq(movie_factors, train_values, rcond=None)

    # 7. Predict preference scores for all movies
    preds = user_vector @ Vt

    # 8. Evaluate the model on the 5 held-out test ratings
    y_true = []
    y_pred = []

    print("\nTest predictions on the extra 5 ratings:\n")
    for movie_id, true_rating in test_ratings.items():
        idx = movie_id_to_index[movie_id]
        pred_rating = preds[idx]

        y_true.append(true_rating)
        y_pred.append(pred_rating)

        title = movie_id_to_title[movie_id]
        print(
            f"{title} | true={true_rating} | predicted={pred_rating:.3f}"
        )

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\nRMSE on the 5 held-out ratings: {rmse:.3f}")

    # 9. Remove all movies the user already rated
    #    so they are not recommended again
    for mid in user_ratings.keys():
        preds[movie_id_to_index[mid]] = -np.inf

    # 10. Sort predictions descending and keep the top 10
    top_indices = np.argsort(preds)[::-1][:10]

    # 11. Show top 10 recommended movies
    print("\nTop 10 recommended movies:\n")
    for rank, idx in enumerate(top_indices, start=1):
        movie_id = movie_ids[idx]
        title = movie_id_to_title.get(movie_id, f"movie_id={movie_id}")
        score = preds[idx]
        print(f"{rank}. {title}  (predicted score: {score:.3f})")


if __name__ == "__main__":
    main()