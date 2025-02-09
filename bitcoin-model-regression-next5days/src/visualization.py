import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(true, pred, title="7-Day Forecast: Predictions vs True Values"):
    """
    Plots the last 100 true values and overlays the past + future predicted values.

    Parameters:
        true (np.array): True values (n_samples, seq_length) up to the prediction point.
        pred (np.array): Predicted values (n_samples, n_future_steps).
        title (str): Title for the plot.
    """
    plt.figure(figsize=(12, 6))

    # Select last 100 samples
    true_values = true # True prices at t=last known
    pred_values = pred[:, -1].flatten()  # Predicted values at t=last known

    # Create time indices (align past and future)
    time_indices_true = np.arange(len(true_values))  # Past true values
    time_indices_pred = np.arange(len(pred_values))  # Past true values
    future_indices = np.arange(len(true_values), len(true_values) + 5)  #### Number of days to predict 📌📌📌

    # Plot true values
    plt.plot(time_indices_true[-250:], true_values[-250:], label='True Prices (Last 250 Days)', color='blue', alpha=0.5, marker='o')

    # Plot past predicted values (aligned with past)
    plt.plot(time_indices_pred[-250:], pred_values[-250:], label='Predicted Prices (Past 250 Days)',color='green', alpha=0.5, marker='o')

    # Plot next 7 predicted values (future)
    future_pred = pred_values[-5:]  ##### Number of days to predict 📌📌📌📌
    #plt.plot(future_indices, future_pred, label='Predicted Prices (Next 7 Days)', linestyle='--', color='red', marker='o')

    # Highlight transition points
    plt.scatter(time_indices_pred[-1], true_values[-1], color='black', label="Prediction Start", alpha = 0.5, zorder=3)
    plt.scatter(future_indices, future_pred, color='red', label="Next 5 days Predictions", zorder=3, alpha = 0.7)

    plt.title("Bitcoin-USD price prediction for the next 5 days")
    plt.xlabel("Time Steps (Days)")
    plt.ylabel("Bitcoin Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()
