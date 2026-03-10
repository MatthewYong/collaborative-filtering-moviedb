import pickle
import random
import numpy as np
from sklearn.metrics import mean_squared_error

MODEL_PATH = "svd_model.pkl"


def load_artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_movies_to_ask(movie_popularity, movie_id_to_index, movie_id_to_title):
    valid_movie_ids = [
        mid for mid in movie_popularity.keys()
        if mid in movie_id_to_index and mid in movie_id_to_title
    ]

    sorted_valid_movie_ids = sorted(
        valid_movie_ids,
        key=lambda mid: movie_popularity[mid],
        reverse=True,
    )

    candidate_pool = sorted_valid_movie_ids[:300]
    random.shuffle(candidate_pool)

    movies_to_ask = [
        (mid, movie_id_to_title[mid])
        for mid in candidate_pool[:100]
    ]

    return movies_to_ask


def get_movies_to_ask():
    artifact = load_artifact()

    movie_ids = artifact["movie_ids"]
    movies_df = artifact["movies_df"]
    movie_popularity = artifact["movie_popularity"]

    movie_id_to_title = dict(zip(movies_df["movie_id"], movies_df["movie_title"]))
    movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    return build_movies_to_ask(
        movie_popularity=movie_popularity,
        movie_id_to_index=movie_id_to_index,
        movie_id_to_title=movie_id_to_title,
    )


def recommend_for_new_user(user_ratings):
    artifact = load_artifact()

    Vt = artifact["Vt"]
    movie_ids = artifact["movie_ids"]
    movies_df = artifact["movies_df"]

    movie_id_to_title = dict(zip(movies_df["movie_id"], movies_df["movie_title"]))
    movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    rated_items = list(user_ratings.items())
    if len(rated_items) < 10:
        raise ValueError("Need at least 10 ratings")

    train_ratings = dict(rated_items[:5])
    test_ratings = dict(rated_items[5:10])

    train_indices = [movie_id_to_index[mid] for mid in train_ratings.keys()]
    train_values = np.array(list(train_ratings.values()), dtype=float)

    movie_factors = Vt[:, train_indices].T
    user_vector, *_ = np.linalg.lstsq(movie_factors, train_values, rcond=None)

    preds = user_vector @ Vt

    y_true = []
    y_pred = []
    test_predictions = []

    for movie_id, true_rating in test_ratings.items():
        idx = movie_id_to_index[movie_id]
        pred_rating = float(preds[idx])

        y_true.append(true_rating)
        y_pred.append(pred_rating)

        test_predictions.append({
            "title": movie_id_to_title[movie_id],
            "true_rating": true_rating,
            "predicted_rating": round(pred_rating, 3),
        })

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    for mid in user_ratings.keys():
        preds[movie_id_to_index[mid]] = -np.inf

    top_indices = np.argsort(preds)[::-1][:10]

    recommendations = []
    for idx in top_indices:
        movie_id = movie_ids[idx]
        recommendations.append({
            "title": movie_id_to_title.get(movie_id, f"movie_id={movie_id}"),
            "score": round(float(preds[idx]), 3),
        })

    return {
        "rmse": round(rmse, 3),
        "test_predictions": test_predictions,
        "recommendations": recommendations,
    }