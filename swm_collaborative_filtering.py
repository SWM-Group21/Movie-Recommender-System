
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

def read_csv_and_print_info(filename):
    df = pd.read_csv(filename)
    print(df.head())
    print(df.info())
    return df

def print_df_details(df, df_name):
    print(f'The {df_name} dataset has', df['userId'].nunique(), 'unique users')
    print(f'The {df_name} dataset has', df['movieId'].nunique(), 'unique movies')
    print(f'The {df_name} dataset has', df['rating'].nunique(), 'unique ratings')
    print(f'The unique {df_name} are', sorted(df['rating'].unique()))
   
ratings=read_csv_and_print_info('ratings.csv')
print_df_details(ratings, 'ratings')

movies = read_csv_and_print_info('movies.csv')
df = pd.merge(ratings, movies, on='movieId', how='inner')
print(df.head())

agg_ratings = df.groupby('title').agg(mean_rating = ('rating', 'mean'),
                                                number_of_ratings = ('rating', 'count')).reset_index()

agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings']>100]
print(agg_ratings_GT100.info())
agg_ratings_GT100.sort_values(by='number_of_ratings', ascending=False).head()

sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_GT100)

df_GT100 = pd.merge(df, agg_ratings_GT100[['title']], on='title', how='inner')
print_df_details(ratings, 'df_GT100')

matrix = df_GT100.pivot_table(index='userId', columns='title', values='rating')
print(matrix.head())

matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 'rows')
print(matrix_norm.head())

user_similarity = matrix_norm.T.corr()
print(user_similarity.head())

user_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))
print(user_similarity_cosine)

def get_ranked_item_score(user_similarity, picked_user_id):
    user_similarity.drop(index=picked_user_id)

    n = 10
    user_similarity_threshold = 0.3
    similar_users = user_similarity[user_similarity[picked_user_id]>user_similarity_threshold][picked_user_id].sort_values(ascending=False)[:n]

    picked_userid_watched = matrix_norm[matrix_norm.index == picked_user_id].dropna(axis=1, how='all')

    similar_user_movies = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')
    similar_user_movies = similar_user_movies.drop(picked_userid_watched.columns,axis=1, errors='ignore')

    item_score = {}

    for i in similar_user_movies.columns:
      movie_rating = similar_user_movies[i]

      total = 0
      count = 0

      for u in similar_users.index:
          if pd.isna(movie_rating[u]) == False:
            score = similar_users[u] * movie_rating[u]
            total += score
            count +=1

      item_score[i] = total / count

    item_score = pd.DataFrame(item_score.items(), columns=['movie', 'movie_score'])

    ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)

    avg_rating = matrix[matrix.index == picked_user_id].T.mean()[picked_user_id]
    ranked_item_score['predicted_rating'] = ranked_item_score['movie_score'] + avg_rating

    ranked_item_score = ranked_item_score.rename_axis('movieId')
    return ranked_item_score

def precision_at_k(predictions, k):
    top_k_predicted = predictions.sort_values(by='predicted_rating', ascending=False).head(k)
    relevant_predicted = top_k_predicted[top_k_predicted['rating'] >= 4.0]
    return len(relevant_predicted) / k if k != 0 else 0

def recall_at_k(predictions, k):
    top_k_predicted = predictions.sort_values(by='predicted_rating', ascending=False).head(k)
    relevant_predicted = top_k_predicted[top_k_predicted['rating'] >= 4.0]
    return len(relevant_predicted) / len(predictions[predictions['rating'] >= 4.0]) if len(predictions[predictions['rating'] >= 4.0])!=0 else 0

def min_max_scaling(merged_df, x):
    return 1 + (x - merged_df['predicted_rating'].min()) * 4 / (merged_df['predicted_rating'].max() - merged_df['predicted_rating'].min())


rmse_values = []
precision_values = []
recall_values = []

def calculate_avg_RMSE(n=30):
    users_30 = ratings['userId'].unique()[:n]
    for picked_user_id in users_30:
        ranked_item_score = get_ranked_item_score(user_similarity, picked_user_id)
        actual_ratings = ratings[ratings['userId']==picked_user_id]
        # Merge predicted ratings DataFrame with actual ratings DataFrame on 'movie'
        merged_df = pd.merge(ranked_item_score, actual_ratings, on='movieId')
        if(merged_df.empty):
            continue
        rmse = np.sqrt(mean_squared_error(merged_df['rating'], merged_df['predicted_rating']))

        top_k = 5
        precision = precision_at_k(merged_df, top_k)
        recall = recall_at_k(merged_df, top_k)

        print(picked_user_id, "->", rmse, precision, recall)
        rmse_values.append(rmse)
        precision_values.append(precision)
        recall_values.append(recall)

num_users = 30
calculate_avg_RMSE(num_users)
avg_rmse = np.mean(rmse_values)
avg_precision = np.mean(precision_values)
avg_recall = np.mean(recall_values)

print("------------------------------------------")
print("Evaluation Metrics")
print("------------------------------------------")
print(f"Average RMSE for the first {num_users} users: {avg_rmse:.4f}")
print(f"Average Precision for the first {num_users} users: {avg_precision:.4f}")
print(f"Average Recall for the first {num_users} users: {avg_recall:.4f}")


# unique_user_ids = ratings['userId'].unique()
# selected_user_ids = unique_user_ids[:num_users]



