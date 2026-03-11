from flask import Flask, render_template, request
from src.recommend import DEFAULT_CONFIG, get_movies_to_ask, recommend_for_new_user

app = Flask(__name__, template_folder="src/templates", static_folder="src/static")


def parse_int(form, key, default):
    value = form.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(form, key, default):
    value = form.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(form, key, default):
    value = form.get(key)
    if value is None:
        return default
    return value.lower() in {"1", "true", "on", "yes"}


def read_config_from_form(form):
    return {
        "n_components": parse_int(form, "n_components", DEFAULT_CONFIG["n_components"]),
        "random_state": parse_int(form, "random_state", DEFAULT_CONFIG["random_state"]),
        "ask_pool_size": parse_int(form, "ask_pool_size", DEFAULT_CONFIG["ask_pool_size"]),
        "ask_count": parse_int(form, "ask_count", DEFAULT_CONFIG["ask_count"]),
        "min_ratings_required": parse_int(
            form,
            "min_ratings_required",
            DEFAULT_CONFIG["min_ratings_required"],
        ),
        "train_rating_count": parse_int(
            form,
            "train_rating_count",
            DEFAULT_CONFIG["train_rating_count"],
        ),
        "test_rating_count": parse_int(
            form,
            "test_rating_count",
            DEFAULT_CONFIG["test_rating_count"],
        ),
        "ridge_alpha": parse_float(form, "ridge_alpha", DEFAULT_CONFIG["ridge_alpha"]),
        "clip_predictions": parse_bool(
            form,
            "clip_predictions",
            DEFAULT_CONFIG["clip_predictions"],
        ),
    }


@app.route("/", methods=["GET"])
def index():
    config = DEFAULT_CONFIG.copy()
    movies = get_movies_to_ask(config=config)
    return render_template("index.html", movies=movies, config=config)


@app.route("/recommend", methods=["POST"])
def recommend():
    user_ratings = {}
    config = read_config_from_form(request.form)

    for key, value in request.form.items():
        if key.startswith("movie_"):
            movie_id = int(key.replace("movie_", ""))
            rating = int(value)

            if rating > 0:
                user_ratings[movie_id] = rating

    if len(user_ratings) < max(
            config["min_ratings_required"],
            config["train_rating_count"] + config["test_rating_count"],
    ):
        return (
            "Need at least "
            f"{max(config['min_ratings_required'], config['train_rating_count'] + config['test_rating_count'])} "
            f"ratings above 0. Currently got {len(user_ratings)}."
        )

    result = recommend_for_new_user(user_ratings, config=config)
    return render_template("results.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)