import numpy as np
import pandas as pd


# Clamps the given value to the 1-5 range
def clamp(x):
    if x > 5:
        return 5
    if x < 1:
        return 1
    return x

def create_svd_predictions():
    # Load data
    users = pd.read_csv("Data/users.csv", sep=";")
    movies = pd.read_csv("Data/movies.csv", sep=";")
    ratings = pd.read_csv("Data/ratings.csv", sep=";")
    predictions = pd.read_csv("Data/predictions.csv", sep=";").values

    # merge users and ratings dataframes on userId column
    df = pd.merge(users, ratings, on='userId')

    # merge movies and merged dataframe on movieId column
    df = pd.merge(movies, df, on='movieId', how='outer')

    # create a pivot table
    pivot_table = df.pivot_table(index='userId', columns='movieId', values='rating')

    # Get the list of existing column labels
    columns = pivot_table.columns

    # Find the missing column labels
    missing_columns = [i for i in range(1, len(columns) + 1) if i not in columns]

    # Insert a new column filled with None values for each missing column
    for i in missing_columns:
        pivot_table.insert(i - 1, i, np.nan)


    # fill the missing values to column averages
    # if column is empty, fill in *4*, otherwise, fill in the mean of the column
    ratings_table = pivot_table.apply(lambda x: x.fillna(4) if x.isnull().all() else x.fillna(x.mean()))

    # convert to numpy array
    ratings_table = np.array(ratings_table)

    # Perform SVD decomposition
    U, S, Vt = np.linalg.svd(ratings_table, full_matrices=False, compute_uv=True)

    # hyperparameter k
    k = 34

    # Take the first k components of U, S, and V
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_k = Vt[:k, :]

    # Reconstruct the matrix using the first k components
    predicted_ratings = U_k @ S_k @ V_k

    # Make predictions
    result = []
    index = 1
    for prediction in predictions:
        movieId = prediction[1] - 1
        userId = prediction[0] - 1
        result.append((index, clamp(predicted_ratings[userId][movieId])))
        index += 1
    return result

# Create submission file
#submission = pd.DataFrame(result, columns=['Id', 'Rating'])
#submission.to_csv('submission.csv', index=False)
