import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense

# Step 1: Create sample user-song interaction data
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3],
    'song_id': [101, 102, 103, 101, 104, 102, 103, 105],
    'liked':   [1, 1, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Step 2: Encode user and song IDs to integers
user_encoder = LabelEncoder()
song_encoder = LabelEncoder()

df['user'] = user_encoder.fit_transform(df['user_id'])
df['song'] = song_encoder.fit_transform(df['song_id'])

num_users = df['user'].nunique()
num_songs = df['song'].nunique()

# Step 3: Build the recommendation model
user_input = Input(shape=(1,))
song_input = Input(shape=(1,))

user_embedding = Embedding(input_dim=num_users, output_dim=8)(user_input)
song_embedding = Embedding(input_dim=num_songs, output_dim=8)(song_input)

user_vec = Flatten()(user_embedding)
song_vec = Flatten()(song_embedding)

dot_product = Dot(axes=1)([user_vec, song_vec])
output = Dense(1, activation='sigmoid')(dot_product)

model = Model([user_input, song_input], output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
model.fit([df['user'], df['song']], df['liked'], epochs=30, verbose=1)

# Step 5: Recommend top songs for a user
def recommend_songs(user_id, top_n=3):
    user_index = user_encoder.transform([user_id])[0]
    all_song_indices = np.arange(num_songs)

    user_array = np.array([user_index] * num_songs)
    predictions = model.predict([user_array, all_song_indices], verbose=0)

    song_scores = list(zip(all_song_indices, predictions.reshape(-1)))
    top_songs = sorted(song_scores, key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\n🎧 Top {top_n} recommendations for User {user_id}:")
    for song_idx, score in top_songs:
        real_song_id = song_encoder.inverse_transform([song_idx])[0]
        print(f"Song ID: {real_song_id} (Score: {score:.2f})")

# Try recommending for User ID 1
recommend_songs(user_id=1)
