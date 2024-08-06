import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('/Users/kwamenaduker/AI/project/crop_yield.csv')
crop_options = data['Crop'].unique()
season_options = data['Season'].unique()

# Streamlit app
st.title("Crop Yield Prediction with Random Forest")

# Crop selection
selected_crop = st.selectbox('Select Crop', crop_options, key='crop')

# Season selection
selected_season = st.selectbox('Select Season', season_options, key='season')

# Filter data based on selected crop and season
filtered_data = data[(data['Crop'] == selected_crop) & (data['Season'] == selected_season)]

# Check if filtered_data is empty
if filtered_data.empty:
    st.write(f"No data available for {selected_crop} during {selected_season}.")
else:
    # Prepare data for Random Forest
    filtered_data['Date'] = pd.to_datetime(filtered_data['Crop_Year'], format='%Y')
    filtered_data = filtered_data.sort_values('Date')
    filtered_data.set_index('Date', inplace=True)
    filtered_data['Yield'] = filtered_data['Yield'].astype(float)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(filtered_data[['Yield']])

    # User input for time frame
    start_year = filtered_data.index.year.min()
    end_year = filtered_data.index.year.max()
    selected_start_year = st.number_input('Start Year', min_value=int(start_year), max_value=int(end_year), value=int(start_year), key='start_year')
    selected_end_year = st.number_input('End Year', min_value=int(start_year), max_value=int(end_year), value=int(end_year), key='end_year')

    # Prepare training data
    def create_sequences(data, seq_length):
        X = []
        y = []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    seq_length = 10  # You can adjust this value
    X, y = create_sequences(scaled_data, seq_length)
    X_train, y_train = X, y

    # Check the shape of X_train to ensure it's correct
    if len(X_train.shape) == 3:
        # X_train is in shape (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
    elif len(X_train.shape) == 2:
        # X_train is already in shape (samples, features)
        pass
    else:
        raise ValueError("Unexpected shape for X_train: {}".format(X_train.shape))

    # Define Random Forest and hyperparameters for Grid Search
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best model from Grid Search
    best_model = grid_search.best_estimator_

    # Save the best model and scaler using joblib
    joblib.dump(best_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Load the model and scaler (in practice, this would be done in a separate session)
    best_model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Predicting future values
    def predict_future(start_year, end_year):
        # Generate future dates
        future_dates = pd.date_range(start=f'{start_year}', end=f'{end_year}', freq='Y')
        future_scaled = scaled_data[-seq_length:]  # Start with the last sequence from training data
        predictions = []

        for _ in range(len(future_dates)):
            pred = best_model.predict(future_scaled.reshape(1, seq_length))
            predictions.append(pred[0])
            future_scaled = np.append(future_scaled[1:], pred)

        # Inverse scale the predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return future_dates, predictions

    # Perform prediction and visualization
    if st.button('Predict'):
        future_dates, future_predictions = predict_future(selected_start_year, selected_end_year)

        # Display the predicted yields
        st.write(f"Predicted yields for {selected_crop} during {selected_season} from {selected_start_year} to {selected_end_year}:")

        # Plotting the predicted values
        fig, ax = plt.subplots()
        ax.plot(future_dates, future_predictions, marker='o', linestyle='-', label='Predicted Yield')
        ax.set_title(f'Predicted Yields for {selected_crop} ({selected_season})')
        ax.set_xlabel('Year')
        ax.set_ylabel('Predicted Yield')
        ax.legend()
        ax.grid(True)

        # Improve spacing and label clarity
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(mdates.YearLocator())  # Set the major tick locator to yearly
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format dates as Year only

        plt.tight_layout()  # Adjust layout to prevent clipping

        # Display the plot in Streamlit
        st.pyplot(fig)
