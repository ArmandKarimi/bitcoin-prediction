 
# src/main.py
import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 👈 Add the parent directory to Python's path so it access modules (e..g config) in the main directory as if it were a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import configuration constants
from config import NAME, START, END, INPUT_SIZE, HIDDEN_SIZE, DROPOUT_RATE, \
                   BATCH_SIZE, EPOCHS, SEQ_LENGTH, PRED_LENGTH, LEARNING_RATE, \
                   D_MODEL,NHEAD, NUM_LAYERS, DIM_FEEDFORWARD,DROPOUT

# Import necessary modules
from fetch_data import load_data
from processing import chronological_split, moving_avg_normalization, create_sequences, data_loader
from model import TimeSeriesTransformer
from train_model import train_model
from test_model import evaluate, inverse_transform
from visualization import plot_predictions
from model_LSTM import Bitcoin_LSTM
from model_GRU import BitcoinGRU
from model_biLSTM import BitcoinBiLSTM

def main():
    # Set device (GPU if available, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # --- Data Fetching ---
    logger.info("Fetching data...")
    data = load_data(NAME, START, END)
    logger.info(f"✅ Data fetched. Shape: {data.shape}")
    
    # --- Data Processing ---
    df_train, df_val, df_test = chronological_split(data)
    logger.info("✅ Data split into train, validation, and test sets.")
    
    X_train = moving_avg_normalization(df_train)
    X_val   = moving_avg_normalization(df_val)
    X_test  = moving_avg_normalization(df_test)
    logger.info("✅ Data normalized.")
    
    X_train_seq, y_train_seq = create_sequences(X_train, SEQ_LENGTH, PRED_LENGTH)
    X_val_seq, y_val_seq     = create_sequences(X_val, SEQ_LENGTH, PRED_LENGTH)
    X_test_seq, y_test_seq   = create_sequences(X_test, SEQ_LENGTH, PRED_LENGTH)
    logger.info(f"✅ Sequences created: Training: {X_train_seq.shape}, Validation: {X_val_seq.shape}, Test: {X_test_seq.shape}")
    
    # Create DataLoaders
    train_loader = data_loader(X_train_seq, y_train_seq, BATCH_SIZE)
    val_loader   = data_loader(X_val_seq, y_val_seq, BATCH_SIZE)
    test_loader  = data_loader(X_test_seq, y_test_seq, BATCH_SIZE)
    
    model = BitcoinGRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # --- Training ---
    logger.info(f"🚀🚀🚀 Training for {EPOCHS} epochs...🚀🚀🚀")
    model = train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, EPOCHS)
    logger.info("✅ Training complete.")
    
    # --- Evaluation ---
    logger.info("Evaluating model...")
    predictions, truths = evaluate(model, test_loader, device, df_test['Close'], window_size=30)
    logger.info("✅ Evaluation complete.")

    #____ reverse normalization_____
    preds = inverse_transform(predictions, df_test['Close'], window_size=30)
    preds = preds.reshape(-1,)
    #trues = inverse_transform(truths, df_test['Close'], window_size=30)

    index = X_test_seq.shape[0]
    trues = df_test['Close'].iloc[-index:].values

    #---MAE----
    mae_error = MAE(preds, trues)
    logger.info(f"📌 MAE = {mae_error:.2f}")

    #---MAE----
    mse_error = MSE(preds, trues)
    logger.info(f"📌 MSE = {np.sqrt(mse_error):.2f}")

    #---- Last True Value ------
    logger.info(f"💵💵💵 True price = {trues[-1]:.2f}")
    
    # --- Decision Logic ---
    today_pred = np.squeeze(preds[-2])
    tomorrow_pred = np.squeeze(preds[-1])
 

    logger.info(f"📊 Predicted for today: {today_pred:.2f}")
    logger.info(f"📊 Predicted for tomorrow: {tomorrow_pred:.2f}")
    
    last_pct_change = (tomorrow_pred - today_pred) * 100 / today_pred
    threshold = 1 

    if last_pct_change < 0:
        logger.warning(f"Alert: Predicted price change is negative: {last_pct_change:.3f}%! Don't buy ⛔")
    elif last_pct_change > threshold:
        logger.warning(f"Alert: Predicted change is {last_pct_change:.3f}% (more than {threshold}%)! Consider buying ✅")
    else:
        logger.warning(f"Alert: Predicted change is {last_pct_change:.3f}%, below threshold {threshold}%. Don't buy	⛔")

    # --- Visualization ---
    plot_predictions(trues, preds, title="Test Set: Predictions vs True Values")

    #---- save model ------
    model_dir = os.path.join("..", "model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))

if __name__ == "__main__":
    main()


