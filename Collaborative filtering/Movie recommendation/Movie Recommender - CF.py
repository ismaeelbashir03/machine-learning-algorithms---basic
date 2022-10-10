# collaborative filtering, using cosign similarity on rows of users with ratings 
# then using the other user with the highest similarity to get the mean ratings
# mulitplied by the similarities as weights to reccomend movies. can be done with movie
# to movie similarities aswell (which we will do).

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# reading in our data and mergin our tables and dropping info we dont want
ratings = pd.read_csv('dataset/ratings.csv')
movies = pd.read_csv('dataset/movies.csv')
ratings = pd.merge(movies, ratings).drop(['genres', 'timestamp'], axis = 1)

# reshaping our dataframe to have an index of userid and columns of the movie 
# name and the values to be the ratings
user_ratings = ratings.pivot_table(index=['userId'], columns=['title'], values='rating')

#removing movies that have less than 10 users who rated it (cleaning our data, as it is very big)
# we also fill our NaN values with zeros
user_ratings = user_ratings.dropna(thresh=10, axis=1).fillna(0)

# now lets standerize our row values, this will give us a mean of 0 and give 
# us a range of 1, (between -1 and 1)
def standerize(row):
    new_row = (row-row.mean())/(row.max()-row.min())
    return new_row

user_ratings = user_ratings.apply(standerize)

# item to item collaborative filtering using cosing similarity
# we are transposing as we are doing an item to item collaborative filter
item_similarity = cosine_similarity(user_ratings.T) # this gives us a similarity matrix from item to item

# we convert it into a data frame
item_similarity = pd.DataFrame(item_similarity, index=user_ratings.columns, columns=user_ratings.columns)

# function to get the similiar movies
def get_similiar_movies(movie_name, user_rating):

    # we are getting the row with that movie name and multiplying the row
    # by the rating the user gave, this will give estimate ratings for the other movies
    # we subtract 2.5 so that if a rating is above 3 it is good, if below it is bad
    similiar_score = item_similarity[movie_name]*(user_rating-2.5)

    # sorting the list by descending, highest at the top
    similiar_score = similiar_score.sort_values(ascending=False)

    return similiar_score

#print(get_similiar_movies('2 Fast 2 Furious (Fast and the Furious 2, The) (2003)', 5))

# testing our algorithm with a user who has given action movies high ratings 
# and romantic movies low ratings and other medium ratings
action_movie_watcher = [
    ('2 Fast 2 Furious (Fast and the Furious 2, The) (2003)', 5),
    ('(500) Days of Summer (2009)', 2),
    ('2012 (2009)', 3),
    ('12 Years a Slave (2013)', 4)
]

# looping through our reviews and getting a list of movies back and appending them to a dataframe
score_total = pd.DataFrame()

for movie, rating in action_movie_watcher:
    # appending the data and ignoring the index as we do not want to append that too
    score_total = score_total.append(get_similiar_movies(movie, rating), ignore_index=True)


# summing all duplicates (gives them a higher score) and using a descending order
score_total = score_total.sum().sort_values(ascending=False)


# dropping the movies that have been already rated from the list (since they most 
# similarly represent themselves)
score_total = score_total.drop([
    action_movie_watcher[0][0], 
    action_movie_watcher[1][0], 
    action_movie_watcher[2][0], 
    action_movie_watcher[3][0]
    ])

#defining a number for our top list
top_num = 10

# printing our reccommendations with their score
print(score_total[:top_num-1])