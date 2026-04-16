import os
import pickle
import random
import numpy as np
from sklearn.metrics import mean_squared_error

MODEL_PATH = "svd_model.pkl"

DEFAULT_CONFIG = {
    "n_components": 20,     # display only -- reflects the trained model's k
    "random_state": 42,     # display only -- used during training, not inference
    "ask_pool_size": 300,
    "ask_count": 100,
    "min_ratings_required": 10,
    "train_rating_count": 5,
    "test_rating_count": 5,
    "ridge_alpha": 0.0,
    "clip_predictions": True,
}

_artifact = None


def _get_artifact():
    """Lazy-load the model artifact. Raises RuntimeError if file is missing."""
    global _artifact
    if _artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model file '{MODEL_PATH}' not found. "
                "Run 'python train_model.py' to train the model first."
            )
        with open(MODEL_PATH, "rb") as f:
            _artifact = pickle.load(f)
    return _artifact


def normalize_config(config=None):
    merged = DEFAULT_CONFIG.copy()
    if config:
        merged.update(config)

    merged["n_components"] = int(merged["n_components"])
    merged["random_state"] = int(merged["random_state"])
    merged["ask_pool_size"] = int(merged["ask_pool_size"])
    merged["ask_count"] = int(merged["ask_count"])
    merged["min_ratings_required"] = int(merged["min_ratings_required"])
    merged["train_rating_count"] = int(merged["train_rating_count"])
    merged["test_rating_count"] = int(merged["test_rating_count"])
    merged["ridge_alpha"] = float(merged["ridge_alpha"])
    merged["clip_predictions"] = bool(merged["clip_predictions"])

    if merged["ask_pool_size"] < 10:
        raise ValueError("ask_pool_size must be at least 10")

    if merged["ask_count"] < 10:
        raise ValueError("ask_count must be at least 10")

    if merged["ask_count"] > merged["ask_pool_size"]:
        raise ValueError("ask_count cannot be larger than ask_pool_size")

    if merged["min_ratings_required"] < 2:
        raise ValueError("min_ratings_required must be at least 2")

    if merged["train_rating_count"] < 1 or merged["test_rating_count"] < 1:
        raise ValueError("train_rating_count and test_rating_count must be at least 1")

    if merged["ridge_alpha"] < 0:
        raise ValueError("ridge_alpha cannot be negative")

    return merged


def build_movies_to_ask(
        movie_popularity,
        movie_id_to_index,
        movie_id_to_title,
        ask_pool_size=300,
        ask_count=100,
):
    valid_movie_ids = [
        mid for mid in movie_popularity.keys()
        if mid in movie_id_to_index and mid in movie_id_to_title
    ]

    sorted_valid_movie_ids = sorted(
        valid_movie_ids,
        key=lambda mid: movie_popularity[mid],
        reverse=True,
    )

    candidate_pool = sorted_valid_movie_ids[:ask_pool_size]
    random.shuffle(candidate_pool)

    movies_to_ask = [
        (mid, movie_id_to_title[mid])
        for mid in candidate_pool[:ask_count]
    ]

    return movies_to_ask


def get_movies_to_ask(config=None):
    artifact = _get_artifact()
    config = normalize_config(config)

    movie_ids = artifact["movie_ids"]
    movies_df = artifact["movies_df"]
    movie_popularity = artifact["movie_popularity"]

    movie_id_to_title = dict(zip(movies_df["movie_id"], movies_df["movie_title"]))
    movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    return build_movies_to_ask(
        movie_popularity=movie_popularity,
        movie_id_to_index=movie_id_to_index,
        movie_id_to_title=movie_id_to_title,
        ask_pool_size=config["ask_pool_size"],
        ask_count=config["ask_count"],
    )


def solve_user_vector(movie_factors, train_values, ridge_alpha=0.0):
    if ridge_alpha > 0:
        xtx = movie_factors.T @ movie_factors
        reg = ridge_alpha * np.eye(xtx.shape[0])
        xty = movie_factors.T @ train_values
        return np.linalg.solve(xtx + reg, xty)

    user_vector, *_ = np.linalg.lstsq(movie_factors, train_values, rcond=None)
    return user_vector


def recommend_for_new_user(user_ratings, config=None):
    artifact = _get_artifact()
    config = normalize_config(config)

    Vt = artifact["Vt"]
    movie_ids = artifact["movie_ids"]
    movies_df = artifact["movies_df"]

    # Override display-only n_components with the actual trained k
    config["n_components"] = artifact["k"]

    movie_id_to_title = dict(zip(movies_df["movie_id"], movies_df["movie_title"]))
    movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    required_total = config["train_rating_count"] + config["test_rating_count"]
    if len(user_ratings) < max(config["min_ratings_required"], required_total):
        raise ValueError(
            f"Need at least {max(config['min_ratings_required'], required_total)} ratings"
        )

    rated_items = list(user_ratings.items())
    train_ratings = dict(rated_items[:config["train_rating_count"]])
    test_ratings = dict(
        rated_items[
        config["train_rating_count"]:
        config["train_rating_count"] + config["test_rating_count"]
        ]
    )

    train_indices = [movie_id_to_index[mid] for mid in train_ratings.keys()]
    train_values = np.array(list(train_ratings.values()), dtype=float)

    # Center the new user's ratings by their own mean before projecting into
    # the SVD space (which was trained on mean-centered data).
    user_mean = float(np.mean(train_values))
    centered_train_values = train_values - user_mean

    movie_factors = Vt[:, train_indices].T
    user_vector = solve_user_vector(
        movie_factors=movie_factors,
        train_values=centered_train_values,
        ridge_alpha=config["ridge_alpha"],
    )

    # Predict in centered space, then add the user's mean back
    centered_preds = user_vector @ Vt
    preds = centered_preds + user_mean

    if config["clip_predictions"]:
        preds = np.clip(preds, 1.0, 5.0)

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
        "config_used": config,
    }