# src/main.py

import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
from torchmetrics.classification import Accuracy  # ✅ Import Accuracy Metric

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 👈 Add the parent directory to Python's path so it access modules (e..g config) in the main directory as if it were a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import configuration constants
from config import NAME, START, END, INPUT_SIZE, HIDDEN_SIZE, DROPOUT_RATE, \
                   BATCH_SIZE, EPOCHS, SEQ_LENGTH, PRED_LENGTH, LEARNING_RATE, NUM_LAYERS

# Import necessary modules
from fetch_data import load_data
from processing import chronological_split, moving_avg_normalization, create_sequences, data_loader
from model_train import train
from model_test import evaluate
from model import model_GRU
from model_LSTM import model_LSTM

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
    
    X_train, y_train = moving_avg_normalization(df_train)
    X_val, y_val   = moving_avg_normalization(df_val)
    X_test, y_test = moving_avg_normalization(df_test)
    logger.info("✅ Data normalized.")
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LENGTH, PRED_LENGTH)
    X_val_seq, y_val_seq     = create_sequences(X_val, y_val, SEQ_LENGTH, PRED_LENGTH)
    X_test_seq, y_test_seq   = create_sequences(X_test, y_test, SEQ_LENGTH, PRED_LENGTH)
    logger.info(f"✅ Sequences created: Training: {X_train_seq.shape}, Validation: {X_val_seq.shape}, Test: {X_test_seq.shape}")
    
    # Create DataLoaders
    train_loader = data_loader(X_train_seq, y_train_seq, BATCH_SIZE)
    val_loader   = data_loader(X_val_seq, y_val_seq, BATCH_SIZE)
    test_loader  = data_loader(X_test_seq, y_test_seq, BATCH_SIZE)
    
    #model = model_GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)
    model = model_LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)
    
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # --- Training ---
    logger.info(f"🚀🚀🚀 Training for {EPOCHS} epochs...🚀🚀🚀")
    model = train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, EPOCHS)
    logger.info("✅ Training complete.")
    
    # --- Evaluation ---
    logger.info("Evaluating model...")
    predictions, truths = evaluate(model, test_loader, device)
    logger.info("✅ Evaluation complete.")


# --- Accuracy Calculation ---
    logger.info("Calculating Accuracy...")

    # Convert predictions to binary labels using Sigmoid + threshold
    sigmoid_preds = torch.sigmoid(torch.tensor(predictions))  # Apply Sigmoid to logits
    preds = (sigmoid_preds > 0.5).float()  # Convert to binary (0 or 1)
    trues = torch.tensor(truths, dtype=torch.float32)  # Ensure truths are float tensor

    preds = preds.to(device)
    trues = trues.to(device)

    # Compute Accuracy
    accuracy_fn = Accuracy(task="binary").to(device)  # Define Accuracy Metric
    accuracy = accuracy_fn(preds, trues)  # Compute Accuracy

    logger.info(f"🚀 🚀 🚀 Model Accuracy: {accuracy:.4f}")

    # ---- True Value ------
    logger.info(f"💵 True price direction <> = {truths[-1].item()}")
    
    # --- Decision Logic ---
    tomorrow_pred = preds[-1].item()  # Extract last prediction (binary)
    today_pred = preds[-2].item()  # Extract previous prediction (binary)

    #logger.info(f"📊 Predicted for tomorrow: {tomorrow_pred[-1]} (1=Up, 0=Down)")
    logger.info(f"📊📊📊 Predicted for today: {today_pred} (1=Up, 0=Down)")

    if tomorrow_pred == 1:
        logger.warning(f"🔔🔔🔔 Alert: Model predicts price will go UP! Consider buying ✅")
    else:
        logger.warning(f"❌❌❌ Alert: Model predicts price will go DOWN! Don't buy ⛔")

    #---- save model ------
    model_dir = os.path.join("..", "model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))


if __name__ == "__main__":
    main()









