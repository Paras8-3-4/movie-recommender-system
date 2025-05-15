import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit config
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# Load and prepare data
@st.cache_data
def load_data():
    file_path = "IMBD.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_path}")
        return pd.DataFrame()

    df.rename(columns={"description": "overview", "genre": "genres"}, inplace=True)

    required_columns = ['overview', 'title', 'genres']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df.dropna(subset=required_columns, inplace=True)
    return df

# Similarity matrix
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    return cosine_similarity(tfidf_matrix)

# Recommendations
def recommend(title, df, sim_matrix):
    idx = df[df['title'].str.lower() == title.lower()].index
    if idx.empty:
        return ["Movie not found."]
    idx = idx[0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    return df['title'].iloc[[i[0] for i in sim_scores]]

# Genre pie chart
def show_movie_genre_pie(movie_title, df):
    movie = df[df['title'].str.lower() == movie_title.lower()]
    if movie.empty:
        st.warning("Movie not found.")
        return

    genres = movie.iloc[0]['genres']
    genre_list = [g.strip() for g in genres.split(',') if g.strip()]
    if not genre_list:
        st.info("No genres available.")
        return

    genre_df = pd.DataFrame({'Genre': genre_list, 'Count': [1]*len(genre_list)})
    fig, ax = plt.subplots()
    ax.pie(
        genre_df['Count'],
        labels=genre_df['Genre'],
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette('pastel')[:len(genre_list)]
    )
    ax.axis('equal')
    st.pyplot(fig)

# Main App
def main():
    st.title("🎬 Smart Movie Recommendation System with Genre Visualizations")

    df = load_data()
    if df.empty:
        st.stop()

    sim_matrix = compute_similarity(df)

    with st.sidebar:
        st.header("🔍 Filter")
        genre_series = df['genres'].str.split(',').explode().str.strip()
        selected_genre = st.selectbox("🎞️ Filter by Genre", ["All"] + sorted(genre_series.dropna().unique()))

    filtered_df = df.copy()
    if selected_genre != "All":
        filtered_df = df[df['genres'].str.contains(selected_genre, case=False, na=False)]

    movie_list = filtered_df['title'].dropna().unique().tolist()
    selected_movie = st.selectbox("🎯 Select a movie to get recommendations:", sorted(movie_list))

    if st.button("🔁 Get Recommendations"):
        results = recommend(selected_movie, df, sim_matrix)

        movie_info = df[df['title'].str.lower() == selected_movie.lower()].iloc[0]
        st.markdown(f"### 🎞️ Selected Movie: **{selected_movie}**")
        st.markdown(f"**Genres:** {movie_info['genres']}")

        st.subheader("🔁 Top 10 Recommended Movies:")
        for movie in results:
            st.markdown(f"- 🎬 **{movie}**")

        st.subheader("📊 Genre Composition of Selected Movie")
        show_movie_genre_pie(selected_movie, df)

    st.markdown("---")
    st.subheader("🎨 Top Genres Across Dataset")

    genre_series_all = df['genres'].str.split(',').explode().str.strip()
    genre_counts = genre_series_all.value_counts().head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="coolwarm", ax=ax)
    ax.set_xlabel("Number of Movies")
    ax.set_title("Top 15 Genres in Dataset")
    st.pyplot(fig)

# Run
if __name__ == '__main__':
    main()
