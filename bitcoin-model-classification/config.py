import datetime

# Yahoo Finance settings
NAME = "BTC-USD"
START = "2014-01-01"
END = datetime.datetime.now()  # Consider using datetime.datetime.now() or "from datetime import datetime"


# Split ratios
RATIOS = (0.7, 0.2, 0.1)

# Sequence and prediction lengths
SEQ_LENGTH = 30
PRED_LENGTH = 1
BATCH_SIZE = 32 # h.param

# Model architecture
INPUT_SIZE = 26
HIDDEN_SIZE = 64
DROPOUT_RATE = 0.3
NUM_LAYERS=2   

# Training parameters
LEARNING_RATE = 0.001
EPOCHS = 50
