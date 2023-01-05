import numpy as np
import pandas as pd


def similarity(row1, row2, ratings):
    # Select the rows as numpy arrays
    row1 = ratings.iloc[row1].to_numpy()
    row2 = ratings.iloc[row2].to_numpy()

    # Calculate the dot product
    dot_product = np.dot(row1, row2)

    # Calculate the norm of each row
    row1_norm = np.linalg.norm(row1)
    row2_norm = np.linalg.norm(row2)
    if row1_norm == 0 or row2_norm == 0:
        return -1

    # Calculate the cosine similarity
    return dot_product / (row1_norm * row2_norm)


def compute_similarity_matrix(ratings, movies):
    matrix = np.zeros((len(movies), len(movies)))
    print(matrix.shape)
    for i in range(0, len(matrix)):
        for j in range(i, len(matrix)):
            matrix[i, j] = similarity(i + 1, j + 1, ratings)
            matrix[j, i] = matrix[i, j]
            print(matrix[i][j])
    return matrix


# Load data
users = pd.read_csv("Data/users.csv", sep=";")
movies = pd.read_csv("Data/movies.csv", sep=";")
ratings = pd.read_csv("Data/ratings.csv", sep=";")
predictions = pd.read_csv("Data/predictions.csv", sep=";").values
print(users)
print(movies)
print(ratings)
print(predictions)

# merge users and ratings dataframes on userId column
df = pd.merge(users, ratings, on='userId')

# merge movies and merged dataframe on movieId column
df = pd.merge(movies, df, on='movieId', how='outer')

# create a pivot table
pivot_table = df.pivot_table(index='userId', columns='movieId', values='rating')

# subtract the mean from each row of the pivot_table
row_means = pivot_table.mean(axis=1)
pivot_table = pivot_table.subtract(row_means, axis=1).fillna(0)
print(pivot_table)

similarity_matrix = compute_similarity_matrix(pivot_table, movies)
sm_df = pd.DataFrame(similarity_matrix)
sm_df.to_csv('similarity_matrix.csv', index=False, header=False)