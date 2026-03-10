from flask import Flask, render_template, request
from src.recommend import get_movies_to_ask, recommend_for_new_user

app = Flask(__name__, template_folder="src/templates", static_folder="src/static")


@app.route("/", methods=["GET"])
def index():
    movies = get_movies_to_ask()
    return render_template("index.html", movies=movies)


@app.route("/recommend", methods=["POST"])
def recommend():
    user_ratings = {}

    for key, value in request.form.items():
        if key.startswith("movie_"):
            movie_id = int(key.replace("movie_", ""))
            rating = int(value)

            if rating > 0:
                user_ratings[movie_id] = rating

    if len(user_ratings) < 10:
        return f"Need at least 10 ratings above 0. Currently got {len(user_ratings)}."

    result = recommend_for_new_user(user_ratings)
    return render_template("results.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)