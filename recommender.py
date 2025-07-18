# recommender.py

import pandas as pd
import numpy as np

# Scikit-learn modules for content-based recommender
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# TensorFlow modules for collaborative filtering
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense

# Global encoders to be reused
user_encoder = LabelEncoder()
song_encoder = LabelEncoder()


# === LOAD AND PREPARE DATA ===
def load_data():
    df = pd.read_csv("songs.csv")
    users = [1, 2, 3]
    liked = [
        [1, 3, 4],  # User 1 likes
        [2, 4, 6],  # User 2 likes
        [1, 5, 7],  # User 3 likes
    ]

    user_song_data = []
    for i, user in enumerate(users):
        for song in liked[i]:
            user_song_data.append((user, song, 1))

    data_df = pd.DataFrame(user_song_data, columns=['user_id', 'song_id', 'liked'])

    # Encode
    data_df['user'] = user_encoder.fit_transform(data_df['user_id'])
    data_df['song'] = song_encoder.fit_transform(data_df['song_id'])

    return df, data_df


# === TENSORFLOW RECOMMENDER ===
def tensorflow_recommender(df, data_df):
    print("\n🔍 Using TensorFlow-based Collaborative Filtering Recommender")

    num_users = data_df['user'].nunique()
    num_songs = data_df['song'].nunique()

    # Model architecture
    user_input = Input(shape=(1,))
    song_input = Input(shape=(1,))
    user_embedding = Embedding(input_dim=num_users, output_dim=8)(user_input)
    song_embedding = Embedding(input_dim=num_songs, output_dim=8)(song_input)

    user_vec = Flatten()(user_embedding)
    song_vec = Flatten()(song_embedding)
    dot = Dot(axes=1)([user_vec, song_vec])
    output = Dense(1, activation='sigmoid')(dot)

    model = Model([user_input, song_input], output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train
    model.fit([data_df['user'], data_df['song']], data_df['liked'], epochs=20, verbose=1)

    # Recommender function
    def recommend(user_id, top_n=3):
        user_index = user_encoder.transform([user_id])[0]
        song_indices = np.arange(num_songs)
        user_array = np.array([user_index] * num_songs)

        preds = model.predict([user_array, song_indices], verbose=0)
        scores = list(zip(song_indices, preds.reshape(-1)))
        top = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

        print(f"\n🎧 Top {top_n} TensorFlow Recommendations for User {user_id}:")
        for song_idx, score in top:
            real_id = song_encoder.inverse_transform([song_idx])[0]
            song_row = df[df['song_id'] == real_id].iloc[0]
            print(f"{song_row['title']} by {song_row['artist']} ({score:.2f})")

    return recommend


# === SCIKIT-LEARN RECOMMENDER ===
def sklearn_recommender(df):
    print("\n📚 Using Scikit-learn-based Content Recommender")

    # Feature Encoding
    features = df[['genre', 'language', 'mood']]
    encoder = OneHotEncoder()
    feature_matrix = encoder.fit_transform(features).toarray()

    similarities = cosine_similarity(feature_matrix)

    def recommend(song_title, top_n=3):
        try:
            idx = df[df['title'].str.lower() == song_title.lower()].index[0]
        except IndexError:
            print("❌ Song not found.")
            return

        scores = list(enumerate(similarities[idx]))
        top = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        print(f"\n🎧 Top {top_n} Similar Songs to '{df.iloc[idx]['title']}':")
        for i, score in top:
            print(f"{df.iloc[i]['title']} by {df.iloc[i]['artist']} ({score:.2f})")

    return recommend


# === MAIN DRIVER ===
import sys

def main():
    # Read mode from command-line argument; default is 'tensorflow'
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "tensorflow"

    df, data_df = load_data()

    if mode == "tensorflow":
        recommender = tensorflow_recommender(df, data_df)
        recommender(user_id=1)

    elif mode == "sklearn":
        recommender = sklearn_recommender(df)
        recommender(song_title="Tum Hi Ho")

    else:
        print("❌ Invalid mode selected. Use 'tensorflow' or 'sklearn'.")


if __name__ == "__main__":
    main()