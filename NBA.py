# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import *
import tensorflow as tf
from statistics import median
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.constraints import MaxNorm
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import itertools
import time
import os

# Get the current working directory
current_directory = os.getcwd()

# Record the start time
start_time = time.time()

# Path to the fantasy odds data
fantasy_odds_folder = os.path.join(current_directory, "./FantasyOdds")


# Load and select variables from PlayerGame CSV files
player_game_list = [pd.read_csv(os.path.join(fantasy_odds_folder, f"PlayerGame.{year}.csv"))[['GameID', 'PlayerID', 'Name', 'Rebounds', 'Turnovers','ThreePointersMade', 'Points', 'Assists', 'Steals', 'BlockedShots']] for year in range(2021, 2025)]
df_player_game = pd.concat(player_game_list)

# Load and select variables from TeamGame CSV files
team_game_list = [pd.read_csv(os.path.join(fantasy_odds_folder, f"TeamGame.{year}.csv"))[['GameID', 'TeamID', 'HomeOrAway', 'Points', 'Assists', 'Rebounds', 'BlockedShots', 'Turnovers', 'Steals']] for year in range(2021, 2025)]
df_team_game = pd.concat(team_game_list)

# Load and select variables from PlayerSeason CSV files
player_season_list = [pd.read_csv(os.path.join(fantasy_odds_folder, f"PlayerSeason.{year}.csv"))[['PlayerID', 'Season', 'Assists', 'Rebounds', 'ThreePointersMade', 'BlockedShots', 'Turnovers', 'Steals']] for year in range(2021, 2025)]
df_player_season = pd.concat(player_season_list)

# Add prefixes to columns to differentiate between different types of statistics
df_games_prefixed = df_games.add_prefix('Game_')
df_player_game_prefixed = df_player_game.add_prefix('PlayerGame_')
df_team_game_prefixed = df_team_game.add_prefix('TeamGame_')
df_player_season_prefixed = df_player_season.add_prefix('PlayerSeason_')

# Remove prefixes from specific columns
columns_to_remove_prefix = ['GameID', 'Season', 'Day', 'AwayTeam', 'HomeTeam', 'AwayTeamID', 'HomeTeamID', 'AwayTeamScore', 'HomeTeamScore', 'PlayerID', 'HomeOrAway', 'TeamID','Name']

for col in columns_to_remove_prefix:
    df_games_prefixed.rename(columns={f'Game_{col}': col}, inplace=True)
    df_player_game_prefixed.rename(columns={f'PlayerGame_{col}': col}, inplace=True)
    df_team_game_prefixed.rename(columns={f'TeamGame_{col}': col}, inplace=True)
    df_player_season_prefixed.rename(columns={f'PlayerSeason_{col}': col}, inplace=True)

# Reset index of each dataframe
df_games_prefixed.reset_index(drop=True, inplace=True)
df_player_game_prefixed.reset_index(drop=True, inplace=True)
df_team_game_prefixed.reset_index(drop=True, inplace=True)
df_player_season_prefixed.reset_index(drop=True, inplace=True)

# Concatenate the dataframes
df_combined = pd.concat([df_games_prefixed, df_player_game_prefixed, df_team_game_prefixed, df_player_season_prefixed], axis=1)

# Replace NaN values with appropriate placeholders (e.g., 0 for statistics that might not be available for all records)
df_combined.fillna(0, inplace=True)


# Merge on common keys
df_combined = pd.merge(df_combined, df_team_game_prefixed, on='TeamID', how='outer')

# Replace NaN values with appropriate placeholders after merge
df_combined.fillna(0, inplace=True)

# Define a function to remove '_x' and '_y' suffixes
def remove_suffix(col):
    if col.endswith('_x'):
        return col[:-2]
    elif col.endswith('_y'):
        return col[:-2]
    else:
        return col

# Apply the function to column names
df_combined.columns = [remove_suffix(col) for col in df_combined.columns]

# Now remove duplicate columns
df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]



df_combined = df_combined.drop_duplicates()



# Splitting the data into features and the labels
X = df_combined[['AwayTeamScore', 'HomeTeamScore', 'TeamGame_Points', 'TeamGame_Assists',
                         'TeamGame_Rebounds', 'TeamGame_BlockedShots', 'TeamGame_Turnovers',
                         'TeamGame_Steals', 'PlayerSeason_Assists', 'PlayerSeason_Rebounds',
                         'PlayerSeason_ThreePointersMade', 'PlayerSeason_BlockedShots',
                         'PlayerSeason_Turnovers', 'PlayerSeason_Steals']].values


y = df_combined[['PlayerGame_Points', 'PlayerGame_Assists', 'PlayerGame_Rebounds',
                 'PlayerGame_ThreePointersMade', 'PlayerGame_BlockedShots',
                 'PlayerGame_Turnovers', 'PlayerGame_Steals']].values

# Define hyperparameters
iterations = 10  # Choose the number of iterations
random_seeds = range(1, iterations)

# Lists to store performance metrics
val_loss_scores = []
best_model_paths = []
mse_loss_scores = []

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(32, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='linear'))  # 7 neurons for 7 target variables

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Evaluate the model
mse_neural = model.evaluate(X_test_scaled, y_test)
print(f"Mean Squared Error: {mse_neural:.3f}")

# Plot the training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



