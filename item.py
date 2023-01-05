from queue import PriorityQueue
import pandas as pd


ratings = pd.read_csv("Data/ratings.csv", sep=";")

predictions = pd.read_csv("Data/predictions.csv", sep=";").values

similarity_matrix = pd.read_csv('similarity_matrix.csv', header=None).values


def make_prediction(user_movies, movieId):
    k = 20
    most_similar = find_most_similar(k, user_movies[:, 1], movieId)
    similiarities_list, movies_list = zip(*most_similar)
    nom = 0.0
    map = {}
    for tup in most_similar:
        map[tup[1]] = -tup[0]
    denom = -sum(similiarities_list)
    for movie in user_movies:
        if movie[1] in movies_list:
            nom += movie[2] * map[movie[1]]
    return nom / denom


def find_most_similar(k, user_movies, movieId):
    pq = PriorityQueue()
    for movie in user_movies:
        pq.put((-similarity_matrix[movie - 1, movieId - 1], movie))
    res = []
    for i in range(k):
        res.append(pq.get())
    return res

def create_item_predictions():
    result = []
    index = 1
    for prediction in predictions:
        movieId = prediction[1]
        userId = prediction[0]
        user_movies = ratings.loc[ratings['userId'] == userId, :].values
        result.append((index, make_prediction(user_movies, movieId)))
        index += 1
    return result


# Create submission file
submission = pd.DataFrame(create_item_predictions(), columns=['Id', 'Rating'])
submission.to_csv('submission-item2-05.csv', index=False)
