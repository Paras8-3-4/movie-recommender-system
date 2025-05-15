# 🎬 Smart Movie Recommendation System

A content-based movie recommendation system built with **Streamlit**, utilizing **TF-IDF vectorization** and **cosine similarity** to suggest similar movies based on plot descriptions. The app also includes **genre-based filtering** and **interactive visualizations** to enhance user experience.

---

## 🔍 Features

- Recommend top 10 movies similar to a selected title.
- Filter recommendations by genre.
- Visualize genre composition of selected movies (Pie Chart).
- View top 15 genres in the dataset (Bar Chart).
- Clean and intuitive UI using Streamlit.

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit
- **Backend:** Python (Pandas, Scikit-learn)
- **Visualization:** Matplotlib, Seaborn
- **Data Source:** IMDb dataset (CSV format)

---

## 📁 Project Structure

```
movie-recommender-system/
│
├── app2.py              # Main Streamlit app
├── IMBD.csv             # Dataset file
├── requirements.txt     # Required Python libraries
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/movie-recommender-system.git
cd movie-recommender-system
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app2.py
```

4. **Open in browser**  
The app will launch in your default browser at `http://localhost:8501`

---

## 📊 Dataset Information

- File: `IMBD.csv`
- Required Columns:
  - `title`
  - `overview` (movie description)
  - `genres` (comma-separated)

---

## 🚀 Future Enhancements

- Integrate collaborative filtering for better personalization
- Add rating-based sorting
- Deploy app to the web (e.g., Streamlit Cloud or Heroku)
- Extend support for TV shows or web series

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/stefanoleone992/imdb-extensive-dataset)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

## 🧑‍💻 Author

**Your Name**  
[GitHub Profile](https://github.com/yourusername)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
