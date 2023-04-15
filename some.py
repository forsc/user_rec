import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

# Example purchase history data
purchases_df = pd.DataFrame({'user_id': ['user1', 'user1', 'user1', 'user2', 'user2', 'user3'],
                             'item_id': ['item1', 'item2', 'item3', 'item1', 'item4', 'item2'],
                             'purchase_date': [datetime(2022, 1, 1), datetime(2022, 2, 1), datetime(2022, 3, 1),
                                               datetime(2022, 1, 1), datetime(2022, 3, 1), datetime(2022, 2, 1)]})

# Function to calculate the similarity between items
def cosine_similarities_matrix(data_matrix):
    similarities = cosine_similarity(data_matrix)
    return similarities

# Function to calculate the similarity between items based on user purchases
def get_item_similarities(user_purchases):
    item_user_matrix = pd.pivot_table(user_purchases, index='item_id', columns='user_id', values='purchase_date', aggfunc='count', fill_value=0)
    item_sim_matrix = cosine_similarities_matrix(item_user_matrix.T)
    return item_sim_matrix

# Function to get similar items based on item similarity matrix and time-based decay
def get_similar_items(item_id, item_sim_matrix, all_items, purchases_df, decay=0.05):
    item_purchases = purchases_df[purchases_df['item_id'] == item_id]
    item_last_purchase_date = max(item_purchases['purchase_date'])
    item_age_days = (datetime.now() - item_last_purchase_date).days
    item_scores = []
    for i in range(len(all_items)):
        item = all_items[i]
        if item == item_id:
            continue
        similarity_score = item_sim_matrix[i][all_items.index(item_id)]
        item_purchases = purchases_df[purchases_df['item_id'] == item]
        item_last_purchase_date = max(item_purchases['purchase_date'])
        item_age_days = (datetime.now() - item_last_purchase_date).days
        item_score = similarity_score * np.exp(-decay*item_age_days)
        item_scores.append((item, item_score))
    return item_scores

# Function to get personalized recommendations for a user
def get_recommendations(user_id, purchases_df, decay=0.05):
    user_purchases = purchases_df[purchases_df['user_id'] == user_id]
    recent_purchase_date = max(user_purchases['purchase_date'])
    recent_purchase_items = list(user_purchases[user_purchases['purchase_date'] == recent_purchase_date]['item_id'])
    item_sim_matrix = get_item_similarities(user_purchases)
    recommended_items = []
    for item in recent_purchase_items:
        similar_items = get_similar_items(item, item_sim_matrix, purchases_df['item_id'].unique().tolist(), purchases_df, decay)
        similar_items = [(item_id, score) for item_id, score in similar_items if item_id not in recent_purchase_items]
        recommended_items.extend(similar_items)
    recommended_items = sorted(recommended_items, key=lambda x: x[1], reverse=True)
    recommended_items = [item_id for item_id, score in recommended_items]
    return recommended_items

# Example usage: Get personalized recommendations for user1 with time-based decay
recommended_items = get_recommendations('user1', purchases_df, decay=0.05
