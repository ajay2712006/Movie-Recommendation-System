import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#t
st.title("Movie Recommendation System")
st.write("Recommend movies based on movie similarity")

#l
movies = pd.read_csv("movies.csv")

# Remove empty rows
movies = movies.dropna()
#sc
movies = movies[["title", "genres", "overview"]]

#cre m info

movies["tags"] = (
    movies["genres"] + " " + movies["overview"]
)
#vec
tfidf = TfidfVectorizer(
    stop_words="english"
)

movie_vectors = tfidf.fit_transform(
    movies["tags"]
)

#f s
similarity = cosine_similarity(
    movie_vectors
)
# s mo
st.subheader("Choose a Movie")

selected_movie = st.selectbox(
    "Select your movie:",
    movies["title"]
)
#recom m

if st.button("Recommend"):

    # Find the movie number
    movie_index = movies[
        movies["title"] == selected_movie
    ].index[0]

    # Get similarity values
    similarity_scores = similarity[movie_index]

    # Sort movies
    similar_movies = list(
        enumerate(similarity_scores)
    )

    similar_movies = sorted(
        similar_movies,
        key=lambda x: x[1],
        reverse=True
    )
    #reco
    st.subheader("Recommended Movies")

    count = 0

    for movie in similar_movies:

        index = movie[0]

        # Don't recommend the same movie
        if index == movie_index:
            continue

        movie_name = movies.iloc[index]["title"]

        st.write(
            str(count + 1) + ". " + movie_name
        )

        count = count + 1

        # Show only 5 movies
        if count == 5:
            break

st.write(
    "This project is made for educational purposes."
)