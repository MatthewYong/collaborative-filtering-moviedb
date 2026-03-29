from flask import Flask, render_template, request
from train_model import (
    run_training,
    DEFAULT_K_CANDIDATES,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_SPLIT,
    DEFAULT_VAL_SPLIT,
    DEFAULT_RANDOM_STATE,
)
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
    # n_components and random_state are training-time params — parsed for
    # display in results.html only; they do not affect SVD inference.
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
    try:
        config = DEFAULT_CONFIG.copy()
        movies = get_movies_to_ask(config=config)
        return render_template("index.html", movies=movies, config=config)
    except RuntimeError as e:
        return str(e), 503


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

    try:
        result = recommend_for_new_user(user_ratings, config=config)
    except RuntimeError as e:
        return str(e), 503

    return render_template("results.html", result=result)


TRAIN_DEFAULTS = {
    "test_split": DEFAULT_TEST_SPLIT,
    "val_split": DEFAULT_VAL_SPLIT,
    "random_state": DEFAULT_RANDOM_STATE,
    "k_candidates": ",".join(str(k) for k in DEFAULT_K_CANDIDATES),
    "model_path": DEFAULT_MODEL_PATH,
}


def parse_k_candidates(raw):
    """Parse a comma-separated string like '10,20,30,50,100' into a sorted int list."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    values = []
    for p in parts:
        try:
            v = int(p)
            if v >= 2:
                values.append(v)
        except ValueError:
            pass
    return sorted(set(values)) if values else DEFAULT_K_CANDIDATES


@app.route("/train", methods=["GET"])
def train_get():
    return render_template("train.html", defaults=TRAIN_DEFAULTS, result=None, error=None)


@app.route("/train", methods=["POST"])
def train_post():
    form = request.form
    params = {
        "test_split": parse_float(form, "test_split", TRAIN_DEFAULTS["test_split"]),
        "val_split": parse_float(form, "val_split", TRAIN_DEFAULTS["val_split"]),
        "random_state": parse_int(form, "random_state", TRAIN_DEFAULTS["random_state"]),
        "k_candidates_raw": form.get("k_candidates", TRAIN_DEFAULTS["k_candidates"]),
        "model_path": form.get("model_path", "").strip() or TRAIN_DEFAULTS["model_path"],
    }
    params["k_candidates"] = parse_k_candidates(params["k_candidates_raw"])

    # Re-populate form values exactly as the user typed them
    displayed = {
        "test_split": params["test_split"],
        "val_split": params["val_split"],
        "random_state": params["random_state"],
        "k_candidates": params["k_candidates_raw"],
        "model_path": params["model_path"],
    }

    try:
        result = run_training(
            model_path=params["model_path"],
            k_candidates=params["k_candidates"],
            test_split=params["test_split"],
            val_split=params["val_split"],
        )
    except Exception as e:
        return render_template(
            "train.html", defaults=displayed, result=None, error=str(e)
        )

    return render_template("train.html", defaults=displayed, result=result, error=None)


if __name__ == "__main__":
    app.run(debug=True)