# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, Model  # Import Model in addition to Sequential
from keras.layers import Dense, Dropout, Input, Concatenate, Flatten, Embedding  # Import necessary layers
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import requests
import shutil
import zipfile
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scikeras.wrappers import KerasRegressor

# Function to download data from the API and save it
def download_data(url, folder_path, filename):
    try:
        response = requests.get(url)
        with open(os.path.join(folder_path, filename), "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False


# Function to extract contents of a zip file
def extract_zip(zip_file, extract_to):
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False


# Function to delete a folder and its contents
def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        return True
    except Exception as e:
        print(f"Error deleting folder: {e}")
        return False


# Function to create a folder
def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
        return True
    except Exception as e:
        print(f"Error creating folder: {e}")
        return False


# Get the current working directory
current_directory = os.getcwd()

# Path to the fantasy odds data
fantasy_odds_folder = os.path.join(current_directory, "FantasyOdds")

# URL to download the data from the API
api_url = "https://sportsdata.io/members/download-file?product=5167afe6-857e-4feb-b0c2-3c239b8ea59c"

# Delete the existing FantasyOdds folder if it exists
if os.path.exists(fantasy_odds_folder):
    delete_folder(fantasy_odds_folder)

# Create a new FantasyOdds folder
if create_folder(fantasy_odds_folder):
    print("FantasyOdds folder created successfully.")
else:
    print("Failed to create FantasyOdds folder. Exiting.")
    exit()

# Download the latest data from the API and save it to the FantasyOdds folder
zip_filename = os.path.join(current_directory, "latest_data.zip")
if download_data(api_url, current_directory, "latest_data.zip"):
    print("Data downloaded successfully.")
    if extract_zip(zip_filename, fantasy_odds_folder):
        print("Data extracted successfully.")
    else:
        print("Failed to extract data. Exiting.")
        delete_folder(fantasy_odds_folder)
        exit()
else:
    print("Failed to download data. Exiting.")
    delete_folder(fantasy_odds_folder)
    exit()

# Player Game data
player_game_list = [pd.read_csv(os.path.join(fantasy_odds_folder, f"PlayerGame.{year}.csv"))[
                        ['Name', 'PlayerID', 'GameID', 'Minutes', 'FieldGoalsMade', 'FieldGoalsAttempted',
                         'ThreePointersMade', 'ThreePointersAttempted', 'FreeThrowsMade', 'FreeThrowsAttempted',
                         'Rebounds', 'Assists', 'Steals', 'BlockedShots', 'Turnovers', 'Points',
                         ]] for year in
                    range(2021, 2025)]
df_player_game = pd.concat(player_game_list)

# Player Game Projection data
player_game_projection_list = [pd.read_csv(os.path.join(fantasy_odds_folder, f"PlayerGameProjection.{year}.csv"))[
                                   ['Name', 'PlayerID', 'GameID', 'Minutes', 'FieldGoalsMade', 'FieldGoalsAttempted',
                                    'ThreePointersMade', 'ThreePointersAttempted', 'FreeThrowsMade',
                                    'FreeThrowsAttempted', 'Rebounds', 'Assists', 'Steals', 'BlockedShots', 'Turnovers',
                                    'Points', 'Day', 'Position', 'InjuryStatus', 'HomeOrAway']] for year in
                               range(2021, 2025)]
df_player_game_projection = pd.concat(player_game_projection_list)

# Factorizing Position/Injury Status and then dropping the Position/Injury Status columns
position_dummies = pd.get_dummies(df_player_game_projection['Position'], prefix='Position')
injury_dummies = pd.get_dummies(df_player_game_projection['InjuryStatus'], prefix='InjuryStatus')
homeoraway_dummies = pd.get_dummies(df_player_game_projection['HomeOrAway'], prefix='HomeOrAway')

# Concatenate the original DataFrame with the dummy variables
df_player_game_projection = pd.concat([df_player_game_projection, position_dummies], axis=1)
df_player_game_projection = pd.concat([df_player_game_projection, injury_dummies], axis=1)
df_player_game_projection = pd.concat([df_player_game_projection, homeoraway_dummies], axis=1)
df_player_game_projection.drop('Position', axis=1, inplace=True)
df_player_game_projection.drop('InjuryStatus', axis=1, inplace=True)
df_player_game_projection.drop('HomeOrAway', axis=1, inplace=True)

df_player_game['PlayerID'] = df_player_game['PlayerID'] % 10000
df_player_game_projection['PlayerID'] = df_player_game_projection['PlayerID'] % 10000
# Convert 'Day' column to datetime
df_player_game_projection['Day'] = pd.to_datetime(df_player_game_projection['Day'], format='%m/%d/%Y %I:%M:%S %p')

# Getting projected data less than today's date
filtered_projection_data = df_player_game_projection[df_player_game_projection['Day'] < pd.Timestamp.today()]

# Merge player game stats with projected stats based on PlayerID and GameID
combined_df = pd.merge(df_player_game, filtered_projection_data, on=['Name', 'PlayerID', 'GameID'],
                       suffixes=('_actual', '_projected'))

# Filter players who have played at least ten minutes in any game
combined_df = combined_df[combined_df['Minutes_actual'] >= 10]

# Step 1: Group the DataFrame by player
grouped_df = combined_df.groupby('PlayerID')

# Step 2: Sort the DataFrame by game date in ascending order
combined_df_sorted = combined_df.sort_values(by=['PlayerID', 'Day'])

# Step 3: Calculate rolling average of points over the last 10 games for each player
combined_df_sorted['AvgPoints_last_10'] = grouped_df['Points_actual'].transform(
    lambda x: x.rolling(window=10, min_periods=1).mean())


output_file_path = "combined_data_sorted.csv"

# New Csv
combined_df_sorted.to_csv(output_file_path, index=False)

# Splitting the data into features and the labels for multiple statistics
X = combined_df[['PlayerID', 'Minutes_projected', 'Points_projected', 'FieldGoalsMade_projected',
                        'ThreePointersMade_projected', 'FreeThrowsMade_projected', 'Rebounds_projected',
                        'Assists_projected', 'Steals_projected', 'BlockedShots_projected', 'Turnovers_projected',
                        'InjuryStatus_Out', 'InjuryStatus_Probable',
                        'InjuryStatus_Questionable',
                        'HomeOrAway_HOME', 'HomeOrAway_AWAY']].values

y = combined_df[['Points_actual']].values


# Define the create_model function with parameters
def create_model(optimizer='adam', dropout_rate=0.3):
    # Define the input layers for other features and player index
    other_features_input = Input(shape=(X_train_scaled.shape[1],), name='OtherFeatures')
    player_index_input = Input(shape=(1,), name='PlayerIndex')

    # Embedding layer for player index
    embedding_dim = 50
    player_embedding_output = Embedding(input_dim=max_player_id, output_dim=embedding_dim)(player_index_input)
    player_embedding_output = Flatten()(player_embedding_output)

    # Concatenate player embedding with other features
    concatenated_input = Concatenate()([player_embedding_output, other_features_input])

    # Define the model architecture
    dense_layer_units = 128
    dense_layer = Dense(dense_layer_units, activation='relu')(concatenated_input)
    dropout_layer = Dropout(dropout_rate)(dense_layer)
    output_layer = Dense(1)(dropout_layer)

    # Define the model with multiple inputs
    model = Model(inputs=[player_index_input, other_features_input], outputs=output_layer)

    # Compile the model
    if optimizer == 'adam':
        optimizer = Adam()
    elif optimizer == 'rmsprop':
        optimizer = RMSprop()
    elif optimizer == 'sgd':
        optimizer = SGD()
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# Define hyperparameters to try
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
epochs = [20, 30, 40]
optimizers = ['adam', 'rmsprop', 'sgd']
dropout_rates = [0.2, 0.3, 0.4]

# Perform hyperparameter tuning
best_score = float('-inf')
best_hyperparameters = {}

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
print("Scaling the data...")
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train[:, 1:])  # Exclude the first column (PlayerID)
X_test_scaled = scaler_X.transform(X_test[:, 1:])  # Exclude the first column (PlayerID)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Reshape y_train_scaled to ensure it's a 1D array
y_train_scaled = y_train_scaled.reshape(-1)

# Determine the maximum player ID
max_player_id = int(np.max(X_train[:, 0])) + 1000  # Adding buffer size

# Reshape player index arrays
X_train_player_index = X_train[:, 0].reshape(-1, 1)
X_test_player_index = X_test[:, 0].reshape(-1, 1)

X_train_other_features = X_train_scaled
X_test_other_features = X_test_scaled

# Ensure both X_train_player_index and X_train_other_features have the same number of samples
assert X_train_player_index.shape[0] == X_train_other_features.shape[0], "Number of samples mismatch"

# Ensure y_train_scaled is a 1D array
y_train_scaled = y_train_scaled.ravel()


# Define the create_model function with best hyperparameters
def create_model():
    # Define the input layers for other features and player index
    other_features_input = Input(shape=(X_train_scaled.shape[1],), name='OtherFeatures')
    player_index_input = Input(shape=(1,), name='PlayerIndex')

    # Embedding layer for player index
    embedding_dim = 50
    player_embedding_output = Embedding(input_dim=max_player_id, output_dim=embedding_dim)(player_index_input)
    player_embedding_output = Flatten()(player_embedding_output)

    # Concatenate player embedding with other features
    concatenated_input = Concatenate()([player_embedding_output, other_features_input])

    # Define the model architecture
    dense_layer_units = 128
    dense_layer = Dense(dense_layer_units, activation='relu')(concatenated_input)
    dropout_layer = Dropout(0.2)(dense_layer)  # Use dropout rate from best hyperparameters
    output_layer = Dense(1)(dropout_layer)

    # Define the model with multiple inputs
    model = Model(inputs=[player_index_input, other_features_input], outputs=output_layer)

    # Compile the model with Adam optimizer and learning rate 0.001
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# Create and train the model with best hyperparameters
keras_model = create_model()
history = keras_model.fit([X_train_player_index, X_train_other_features], y_train_scaled,
                          epochs=40, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
print("Evaluating the model...")
test_loss = keras_model.evaluate([X_test_player_index, X_test_other_features], y_test_scaled)
print("Test Loss:", test_loss)


# # Create the Keras model
# keras_model = create_model()
#
# # Train the model
# print("Training the model...")
# history = keras_model.fit([X_train_player_index, X_train_other_features], y_train_scaled, epochs=23, batch_size=87,
#                           validation_split=0.2, verbose=1)
#
# # Evaluate the model
# print("Evaluating the model...")
# test_loss = keras_model.evaluate([X_test_player_index, X_test_other_features], y_test_scaled)
#
# # Print the test loss
# print("Test Loss:", test_loss)
#
#


# Define a function to predict points for players projected to play today
def predict_player_stat_lines():
    # Get today's date
    today_date = pd.Timestamp.today().date()

    # Filter the DataFrame to get the projected stats for players projected to play today
    projected_players_today = df_player_game_projection[
        (df_player_game_projection['Day'].dt.date == today_date) &
        (df_player_game_projection['Minutes'] > 10)]

    # Initialize an empty DataFrame to store the predicted stats
    predicted_stats_df = pd.DataFrame(columns=['Player', 'Predicted Points'])

    # Loop through each player projected to play today
    for player_name in projected_players_today['Name'].unique():
        try:
            # Get the last instance of the player's AvgPoints_last_10 from sorted combined_df
            player_last_stats = combined_df_sorted[combined_df_sorted['Name'] == player_name].tail(1)

            # Extract the last instance of AvgPoints_last_10
            last_avg_points_last_10 = player_last_stats['AvgPoints_last_10'].values[0]

            # Append last_avg_points_last_10 to player_stats_today
            player_stats_today = projected_players_today[projected_players_today['Name'] == player_name]
            player_stats_today['AvgPoints_last_10'] = last_avg_points_last_10

            # Extract the projected stats columns
            projected_stats = player_stats_today.iloc[0][['PlayerID', 'Minutes', 'Points', 'FieldGoalsMade',
                                                          'ThreePointersMade', 'FreeThrowsMade', 'Rebounds',
                                                          'Assists', 'Steals', 'BlockedShots', 'Turnovers',
                                                          'InjuryStatus_Out',
                                                          'InjuryStatus_Probable',
                                                          'InjuryStatus_Questionable',
                                                          'HomeOrAway_HOME', 'HomeOrAway_AWAY']].values

            # Scale the projected stats
            scaled_projected_stats = scaler_X.transform(projected_stats[1:].reshape(1, -1))  # Exclude PlayerID

            # Find the player index
            player_index = combined_df[combined_df['Name'] == player_name]['PlayerID'].iloc[0]

            # Use the trained model to predict the player's actual stat line
            predicted_stats_scaled = keras_model.predict(
                [np.array([[player_index]]), scaled_projected_stats])

            predicted_stats = scaler_y.inverse_transform(predicted_stats_scaled)

            # Append the predicted points to the DataFrame
            predicted_stats_df = predicted_stats_df.append({'Player': player_name,
                                                            'Predicted Points': predicted_stats[0][0]},
                                                           ignore_index=True)
        except IndexError:
            print("Player not found or data unavailable. Please check the player's name.")

    # Save the predicted stats to a CSV file
    predicted_stats_df.to_csv('predicted_player_stats.csv', index=False)
    print("Predicted player stats saved to predicted_player_stats.csv")


# Call the function to predict player stat lines
predict_player_stat_lines()

