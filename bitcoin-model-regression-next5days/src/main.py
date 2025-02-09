 
# src/main.py
import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
from sklearn.metrics import mean_squared_error as mse
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
    
    # --- Model Setup ---
    # model = TimeSeriesTransformer(
    #     input_size=INPUT_SIZE,
    #     d_model=D_MODEL,        
    #     nhead=NHEAD,            
    #     num_layers=NUM_LAYERS,       
    #     dim_feedforward=DIM_FEEDFORWARD, 
    #     dropout=DROPOUT
    # )
    

    model = Bitcoin_LSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout_rate = DROPOUT_RATE)
    #model = BitcoinGRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE)
    #model = BitcoinBiLSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, dropout_rate = DROPOUT)


    model.to(device)
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss(beta=0.05)  # Adjust beta as needed
    #criterion = HypercosineLoss(alpha=0.2)
    #criterion = nn.SmoothL1Loss(beta=1.0)  # Adjust beta as needed
    #criterion = nn.CosineEmbeddingLoss()


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
    #trues = inverse_transform(truths, df_test['Close'], window_size=30)

    index = X_test_seq.shape[0]
    trues = df_test['Close'].iloc[-index:].values

    print("trues", trues[0])
    print("trues shape", trues.shape)
    print("preds shape", preds[:, -1].flatten().shape)



    #---MSE----
    # mae_error = MAE(preds[-2], trues[-1])
    # logger.info(f"📌 MAE = {mae_error:.2f}")

    #---- True Value ------
    #logger.info(f"💵💵💵 True price = {trues[-1]:.2f}")
    
    # --- Decision Logic ---
    logger.info(f"📊 The last 7 days: {trues[-7:]}")
    logger.info(f"📊 📊 The next 7 days: {preds[-1]}")

    # --- Visualization ---
    plot_predictions(trues, preds, title="Test Set: Predictions vs True Values")

if __name__ == "__main__":
    main()
