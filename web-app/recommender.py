import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
# import matplotlib.pyplot as plt
from imdb import Cinemagoer
ia = Cinemagoer()


def session_begin():
	for i in range(num_movies):
		m_vector = [0 for i in range(len(genre_dict))]
		genres = movies_df.iloc[i]['genres'].split('|')
		m_vecor = make_vec_preference(genres, m_vector)
		m_vector = torch.tensor(m_vector)
		m_vector = unify_vec(m_vector)
		m_dict.append([movies_df.iloc[i]['movieId'], m_vector])
	vector_list = []
	for data in m_dict:
		vector_list.append(data[1].tolist())
	
	st.session_state['neighbors'] = NearestNeighbors(n_neighbors=k, algorithm='auto')
	st.session_state['neighbors'].fit(vector_list)
	st.session_state['final_df'] = None
	st.session_state['links_df'] = pd.read_csv('pages/links.csv', dtype = {'imdbId': str, 'movieId': int})





st.button('Start Session', on_click=session_begin)


st.write("# Movie Recommender System")

st.write("This is a demonstration of a simple content based recommendation system. The model uses K Nearest Neighbors search to find out the movies that are most suited to the user's genre of interest.")


movies_df = pd.read_csv('pages/movies_df.csv')
ratings_df = pd.read_csv('pages/ratings_df.csv')

preferences = st.multiselect(
    'Select/Type in some movies',
    movies_df['title'])

preferences = movies_df.loc[movies_df['title'].isin(preferences)]['index'].tolist()

# k = st.number_input("How many recommendations do you want?", value=10)
k = 10






# Use this section to display insights about the preferences
# if len(preferences) > 0:
# 	# st.write('Your preference profile:')
# 	genre_list = movies_df.loc[movies_df['index'].isin(preferences)]['genres'].tolist()
# 	genre_map = {}
# 	total_genres = 0
# 	for genres in genre_list:
# 		for genre in genres.split('|'):
# 			total_genres = total_genres + 1
# 			if genre not in genre_map:
# 				genre_map[genre] = 1
# 			else:
# 				genre_map[genre] = genre_map[genre] + 1
	# for k,v in genre_map.items():
	# 	st.write(k + ' : ' + str("{:.2f}".format((v/total_genres)*100))+'%')
	# labels = list(genre_map.keys())
	# sizes = list(genre_map.values())
	# explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
	# fig1, ax1 = plt.subplots()
	# ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=False, startangle=90)
	# ax1.axis('equal')
	# st.pyplot(fig1)



print('PREFERENCES')
print(preferences)
m_dict = []
num_movies = len(movies_df) 
num_users = len(ratings_df['userId'].unique())

genre_dict = {
	'Action': 0,
    'Adventure': 1,
    'Animation': 2,
    "Children": 3,
    'Comedy': 4,
    'Crime': 5,
    'Documentary': 6,
    'Drama': 7,
    'Fantasy': 8,
    'Film-Noir': 9,
    'Horror': 10,
    'Musical': 11,
    'Mystery': 12,
    'Romance': 13,
    'Sci-Fi': 14,
    'Thriller': 15,
    'War': 16,
    'Western': 17}



def unify_vec(m_vector):
  if np.sqrt(torch.matmul(m_vector, m_vector)) != 0:
    return m_vector/np.sqrt(torch.matmul(m_vector, m_vector))
  return m_vector

def make_vec_preference(genres, m_vector):
	for genre in genres:
		if genre in genre_dict.keys():
			m_vector[genre_dict[genre]] = 1
	return m_vector


def generate_recommendations():
	st.session_state['final_df'] = None
	for preference in preferences:
		query_vector = [0 for i in range(len(genre_dict))]
		rec = movies_df.loc[movies_df['index'] == preference]
		for genre in rec.iloc[0][3].split('|'):
			if genre in genre_dict.keys():
				query_vector[genre_dict[genre]] = query_vector[genre_dict[genre]] + 1
		query_vector = unify_vec(torch.tensor(query_vector)).tolist()
		distances, indices = st.session_state['neighbors'].kneighbors([query_vector])
		if st.session_state['final_df'] is not None:
			st.session_state['final_df'] = pd.concat([st.session_state['final_df'], movies_df.loc[ movies_df.index.isin(indices[0]) & (~movies_df.index.isin(preferences))].sort_values(by='avg_rating', ascending=False)])
		else:
			st.session_state['final_df'] = movies_df.loc[ movies_df.index.isin(indices[0]) & (~movies_df.index.isin(preferences))].sort_values(by='avg_rating', ascending=False)


st.button('Suggest movies!', on_click=generate_recommendations)


if 'final_df' in st.session_state and st.session_state['final_df'] is not None:
	# st.write(st.session_state['final_df'])
	for i in range(len(st.session_state['final_df'])):
		try:
			title = st.session_state['final_df'].iloc[i]['title']
			genres = st.session_state['final_df'].iloc[i]['genres'].replace('|', ' ')
			mov_id = st.session_state['final_df'].iloc[i]['movieId']
			imdb_id = st.session_state['links_df'].loc[st.session_state['links_df']['movieId'] == mov_id+1].iloc[0]['imdbId']
			st.markdown("""---""")
			st.image(
	            str(ia.get_movie(imdb_id)['cover']),
	            width=100, # Manually Adjust the width of the image as per requirement
	        )
			st.write("Title: " +title)
			st.write("Genres: " + genres)
			st.markdown("""---""")
		except:
			print('Could not fetch movie')
