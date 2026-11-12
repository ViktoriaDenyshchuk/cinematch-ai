
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def load_movies():
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    links = pd.read_csv(
    DATA_DIR / "links.csv",
    dtype={"imdbId": "string", "tmdbId": "string"})

    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={
            "mean": "avg_rating",
            "count": "rating_count"
        })
    )

    movies = movies.merge(stats, on="movieId", how="left")
    movies["avg_rating"] = movies["avg_rating"].fillna(0)
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
    
    movies = movies.merge(links, on="movieId", how="left")
    
    movies["imdb_url"] = movies["imdbId"].apply(
    lambda x: f"https://www.imdb.com/title/tt{str(x).zfill(7)}/"
    if pd.notna(x) else None)
   
    movies["tmdb_url"] = movies["tmdbId"].apply(
    lambda x: f"https://www.themoviedb.org/movie/{x}"
    if pd.notna(x) and str(x).strip() else None)

    return movies


MOVIES = load_movies()


def search_movies(
    genre: str = "",
    min_rating: float = 0,
    year_from: int = 1900,
    limit: int = 5
):
    df = MOVIES.copy()

    if genre:
        df = df[
            df["genres"].str.contains(
                genre, case=False, regex=False, na=False
            )
        ]

    if year_from:
        years = pd.to_numeric(
            df["title"].str.extract(r"\((\d{4})\)\s*$")[0],
            errors="coerce"
        )
        df = df[years >= year_from]

    df = df[
        (df["avg_rating"] >= min_rating)
        & (df["rating_count"] >= 5)
    ]

    df = df.sort_values(
        ["avg_rating", "rating_count"],
        ascending=[False, False]
    )

    return df.head(min(max(limit, 1), 20))[
        ["title", "genres", "avg_rating", "rating_count",  "imdb_url", "tmdb_url"]
    ].round(2).to_dict(orient="records")


def compare_movies(title1: str, title2: str):
    results = []

    for title in [title1, title2]:
        matches = MOVIES[
            MOVIES["title"].str.contains(
                title, case=False, regex=False, na=False
            )
        ]

        if matches.empty:
            results.append({
                "query": title,
                "error": "Фільм не знайдено у MovieLens"
            })
            continue

        matches = matches.sort_values(
            "rating_count", ascending=False
        )

        movie = matches.iloc[0]

        results.append({
            "query": title,
            "title": movie["title"],
            "genres": movie["genres"],
            "avg_rating": round(float(movie["avg_rating"]), 2),
            "rating_count": int(movie["rating_count"]),
            "imdb_url": movie["imdb_url"],
            "tmdb_url": movie["tmdb_url"]
        })

    return results
