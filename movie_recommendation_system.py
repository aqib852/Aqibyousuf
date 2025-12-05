"""
=========================================================
🎬 MOVIE RECOMMENDATION SYSTEM
Using TF-IDF Vectorization + Cosine Similarity
=========================================================

This project recommends movies based on similarity
of movie descriptions and genres.

Concepts Used:
✅ Pandas
✅ TF-IDF Vectorization
✅ Cosine Similarity
✅ Content-Based Recommendation System

Author: Your Name
=========================================================
"""

# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# 2. CREATE MOVIE DATASET
# -------------------------------

# You can replace this with a real dataset later
data = {
    "title": [
        "Inception", "Interstellar", "Titanic",
        "Avengers", "Joker", "Iron Man",
        "The Dark Knight", "Gravity"
    ],
    "overview": [
        "Dream inside dream thriller",
        "Space travel to save humanity",
        "Love story on a sinking ship",
        "Superheroes save the world",
        "A man becomes a dangerous criminal",
        "A genius builds a powerful suit",
        "Batman fights a dangerous villain",
        "Astronaut trapped in space"
    ],
    "genre": [
        "Sci-Fi", "Sci-Fi", "Romance",
        "Action", "Drama", "Action",
        "Action", "Sci-Fi"
    ]
}

movies = pd.DataFrame(data)


# -------------------------------
# 3. COMBINE IMPORTANT TEXT
# -------------------------------

movies["combined"] = movies["overview"] + " " + movies["genre"]


# -------------------------------
# 4. VECTORIZE TEXT USING TF-IDF
# -------------------------------

vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(movies["combined"])


# -------------------------------
# 5. APPLY COSINE SIMILARITY
# -------------------------------

similarity = cosine_similarity(vectors)


# -------------------------------
# 6. RECOMMENDATION FUNCTION
# -------------------------------

def recommend(movie_name):
    if movie_name not in movies["title"].values:
        print("\n❌ Movie not found in database!")
        return

    movie_index = movies[movies["title"] == movie_name].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\n✅ Recommended Movies for:", movie_name)
    print("------------------------------------")
    for i in movie_list:
        print(movies.iloc[i[0]].title)


# -------------------------------
# 7. USER INPUT
# -------------------------------

print("\n🎬 MOVIE RECOMMENDATION SYSTEM")
print("---------------------------------")
print("Available Movies:")
print(movies["title"].to_string(index=False))

movie_name = input("\nEnter a movie name: ")

recommend(movie_name)
