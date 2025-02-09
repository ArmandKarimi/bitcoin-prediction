import matplotlib.pyplot as plt

def plot_predictions(true, pred, title="Predictions vs True Values"):
    """
    Plots true vs. predicted values for the first predicted time step.
    
    Parameters:
        true (np.array): Array of true values with shape (n_samples, n_predictions).
        pred (np.array): Array of predicted values with shape (n_samples, n_predictions).
        title (str): Title for the plot.
        
    Note:
        The y-axis label is set to "Bitcoin Price (USD)". If your predictions are
        percent changes (Close), consider changing it to "Percent Change (%)" or
        another appropriate label.
    """
    plt.figure(figsize=(12, 6))
    # plt.plot(true[:, 0], label='True prices', alpha=0.7)
    # plt.plot(pred[:, 0], label='Predicted prices', linestyle='--')
    plt.plot(true, label='True prices', alpha=0.7)
    plt.plot(pred, label='Predicted prices', linestyle='--')
    plt.title("Bitcoin price prediction USD")
    plt.xlabel("Samples")
    plt.ylabel("Bitcoin Price (USD)")
    plt.legend()
    plt.show()
