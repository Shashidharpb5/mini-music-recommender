# 🎧 Mini Music Recommendation System

A simple music recommendation engine using both **TensorFlow** and **Scikit-learn**.  
This project demonstrates two approaches to building a song recommendation system:

- A **deep learning-based model** using collaborative filtering via **TensorFlow embeddings**
- A **content-based model** using **Scikit-learn** similarity techniques

✅ CLI support included for easy testing and interaction.


## 🚀 Features

- 🔍 Content-based filtering with cosine similarity (Scikit-learn)
- 🤖 Deep learning model with user-item embeddings (TensorFlow)
- 🧠 Learns song preferences and makes recommendations
- ⚙️ Simple command-line interface for switching between methods


## ⚙️ Modes Comparison

| Feature                     | TensorFlow (Collaborative) | Scikit-learn (Content-based) |
|----------------------------|-----------------------------|-------------------------------|
| Technique Used             | User-Song Embeddings        | TF-IDF + Cosine Similarity    |
| Input                      | User ID                     | Song Title                    |
| Output                     | Recommended Song IDs        | Similar Songs (Title + Score) |
| Suitable For               | Large user-item datasets    | Cold-start / metadata usage   |



## 🛠️ Tech Stack

- 🐍 Python
- 🧠 TensorFlow & Keras
- 🧮 Scikit-learn
- 📊 Pandas, NumPy



## 📦 Installation

### 1️⃣ Clone the repo:


git clone https://github.com/your-username/mini-music-recommender.git
cd mini-music-recommender

## 2️⃣ Create & activate virtual environment:

python -m venv venv
venv\Scripts\activate  # On Windows

## 3️⃣ Install dependencies:

pip install -r requirements.txt

## ▶️ Run the Project
You can switch between TensorFlow or Scikit-learn mode using the mode variable inside recommender.py.


# === MAIN DRIVER ===
def main():
    mode = "sklearn"  # Change to "tensorflow" as needed

Then run:
python recommender.py

## 📸 Sample Output

   ## 📚 Using Scikit-learn-based Content Recommender

🎧 Top 3 Similar Songs to 'Tum Hi Ho':
Agar Tum Saath Ho by Alka Yagnik (1.00)
Dil Diyan Gallan by Atif Aslam (0.67)
Shape of You by Ed Sheeran (0.00)

## Or for TensorFlow:
   ## 🤖 Using TensorFlow-based Collaborative Recommender

🎧 Top 3 recommendations for User 1:
Song ID: 101 (Score: 0.51)
Song ID: 105 (Score: 0.49)
Song ID: 103 (Score: 0.48)


## 📁 Folder Structure

mini-music-recommender/
├── recommender.py             # Main CLI + logic controller
├── requirements.txt           # Project dependencies
├── songs.csv                  # Song metadata
├── user_song_interactions.csv # User-song interaction matrix
├── README.md                  # Project overview
├── .gitignore
└── venv/                      # Virtual environment


## 🤝 Author

**Pinnamshetty Shashidhar**
📧 [228r1a66b5@gmail.com](mailto:228r1a66b5@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/shashidharpb5) • [GitHub](https://github.com/shashidharpb5)