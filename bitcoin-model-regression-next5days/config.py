import datetime

# Yahoo Finance settings
NAME = "BTC-USD"
START = "2014-01-01"
END = datetime.datetime.now()  # Consider using datetime.datetime.now() or "from datetime import datetime"


# Split ratios
RATIOS = (0.5, 0.3, 0.2)

BUFFER = 10
# Sequence and prediction lengths
SEQ_LENGTH = 30
PRED_LENGTH = 5 #### number of days to predict 📌📌📌📌

# Model architecture
INPUT_SIZE = 25 
HIDDEN_SIZE = 32
DROPOUT_RATE = 0.4

NUM_LAYERS=2   
DROPOUT=0.3

# Transformer Architecture
D_MODEL = 32       
NHEAD= 4           
DIM_FEEDFORWARD= 32


# Training parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64

EPOCHS = 40
