import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.title("Movie Recommendation System")
st.write("Recommend movies based on movie similarity")
movies = pd.read_csv("movies.csv")

movies = movies.dropna()

movies = movies[["title", "genres", "overview"]]

movies["tags"] = (
    movies["genres"] + " " + movies["overview"]
)

tfidf = TfidfVectorizer(
    stop_words="english"
)

movie_vectors = tfidf.fit_transform(
    movies["tags"]
)

similarity = cosine_similarity(
    movie_vectors
)
st.subheader("Choose a Movie")

selected_movie = st.selectbox(
    "Select your movie:",
    movies["title"]
)


if st.button("Recommend"):
    movie_index = movies[
        movies["title"] == selected_movie
    ].index[0]
    similarity_scores = similarity[movie_index]

    similar_movies = list(
        enumerate(similarity_scores)
    )

    similar_movies = sorted(
        similar_movies,
        key=lambda x: x[1],
        reverse=True
    )
    st.subheader("Recommended Movies")

    count = 0

    for movie in similar_movies:

        index = movie[0]

       if index == movie_index:continue

        movie_name = movies.iloc[index]["title"]

        st.write(
            str(count + 1) + ". " + movie_name
        )

        count = count + 1

     
        if count == 5:
            break

st.write(
    "This project is made for educational purposes."
)
