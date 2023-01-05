# Reccomender System
This project is designed to predict movie ratings for a given set of users and movies. Two different prediction methods are implemented: Singular Value Decomposition (SVD) and item-based collaborative filtering.

### Prerequisites
This project requires Python 3 and the following libraries:

NumPy <br>
pandas
### Usage
To use the prediction methods, first run compute the similarity matrix in the similarity-matrix. Import the svd and item modules and call the create_svd_predictions() and create_item_predictions() functions, respectively. These functions will return lists of tuples containing the prediction ID and the predicted rating.

To combine the predictions using the provided weights, use the combine_predictions() function from main.py module. This function takes in two lists of predictions and the weights for each method and returns a single list of combined predictions.

To create a CSV file of the predictions, use the create_submission_file() function. This function takes in a list of predictions and a file name, and it will save the predictions to a CSV file with the given file name.

### Data
The data used in this project consists of three CSV files:
<li>users.csv: contains user information such as age and occupation
<li>movies.csv: contains movie information such as title and genre
<li>ratings.csv: contains user-movie ratings

In addition, the program is run on a predictions.csv file, which contains user and movie ids for which the ratings should be predicted.

### Methodology

Singular Value Decomposition (SVD)
The SVD method decomposes the user-movie rating matrix into three matrices: a matrix of user vectors, a diagonal matrix of singular values, and a matrix of movie vectors. The predicted ratings are obtained by reconstructing the rating matrix using a reduced number of singular values and vectors.

Item-based Collaborative Filtering
The item-based approach uses the ratings of similar movies to make predictions for a given movie. The similarity between movies is calculated using the Pearson correlation coefficient and stored in a similarity matrix. To make a prediction for a given user and movie, the ratings of the most similar movies to the target movie that the user has rated are averaged.

Combining the results:
The results are combined using a weighted average of the results computed by the methods above.
### Authors
Marcin Jarosz - Initial work
