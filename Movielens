
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Sample ratings data (simulated MovieLens-style dataset)
ratings_data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
    'movie_id': [10, 20, 30, 10, 30, 20, 30, 10, 20, 40],
    'rating': [4, 5, 3, 5, 2, 4, 4, 3, 5, 5],
}
ratings_df = pd.DataFrame(ratings_data)

# Create user-item matrix
user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Recommendation function
def recommend_movies(user_id, num_recommendations=2):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    weighted_scores = np.dot(similar_users.values, user_item_matrix.loc[similar_users.index])
    recommendations = pd.Series(weighted_scores, index=user_item_matrix.columns)
    already_rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = recommendations.drop(index=already_rated, errors='ignore')
    return recommendations.sort_values(ascending=False).head(num_recommendations)

# Get recommendations for a user
user_id = 1
print(f"Recommendations for User {user_id}:")
print(recommend_movies(user_id))

# Plot and save user similarity heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(user_similarity_df, annot=True, cmap='Blues', xticklabels=True, yticklabels=True)
plt.title("User Similarity Matrix (Cosine Similarity)")
plt.tight_layout()
plt.savefig("user_similarity_heatmap.png")
plt.show()
