# Bitcoin Prediction Project

This project contains multiple machine learning models designed to predict Bitcoin price movements and trends using various approaches, such as classification and regression. Each model is dockerized and can be deployed independently.

---

## **Project Structure**
```
📂 bitcoin-prediction/
│
├── 📂 bitcoin-model-classification/
│   ├── 📂 src/
│   │   ├── __init__.py       # Marks the folder as a Python package
│   │   ├── main.py           # Script for model inference
│   │   ├── other scripts     # Supporting scripts (if any)
│   ├── requirements.txt      # Python dependencies for classification
│   ├── Dockerfile            # Docker configuration for classification
│   ├── README.md             # Details specific to this model
│
├── 📂 bitcoin-model-regression/
│   ├── 📂 src/
│   │   ├── __init__.py       # Marks the folder as a Python package
│   │   ├── main.py           # Script for regression inference
│   │   ├── other scripts     # Supporting scripts (if any)
│   ├── requirements.txt      # Python dependencies for regression
│   ├── Dockerfile            # Docker configuration for regression
│   ├── README.md             # Details specific to this model
│
├── 📂 bitcoin-model-regression_3days/
│   ├── 📂 src/
│   │   ├── __init__.py       # Marks the folder as a Python package
│   │   ├── main.py           # Script for 3-day regression inference
│   │   ├── other scripts     # Supporting scripts (if any)
│   ├── requirements.txt      # Python dependencies for 3-day regression
│   ├── Dockerfile            # Docker configuration for 3-day regression
│   ├── README.md             # Details specific to this model
│
├── .gitignore                # Files and folders to ignore in Git
├── README.md                 # General project documentation
└── other project-level files (if needed)
```

---

## **Models**
### 1. **Classification**
- Predicts if the Bitcoin price will rise or fall.
- Contains a Dockerized LSTM classifier.

### 2. **Regression**
- Predicts the future Bitcoin price for 1-day intervals.
- Includes a regression model (GRU)

### 3. **3-Day Regression**
- Predicts Bitcoin prices for the next 3 days.
- Includes a specialized regression model (LSTM)

---

## **Setup Instructions**

### 1. **Clone the Repository**
```bash
git clone https://github.com/ArmandKarimi/bitcoin-prediction
cd bitcoin-prediction
```

### 2. **Set Up a Python Virtual Environment**
```bash
python3 -m venv env
source env/bin/activate
```

### 3. **Install Dependencies**
Navigate to the model folder you want to run (e.g., `bitcoin-model-classification`) and install its dependencies:
```bash
cd bitcoin-model-classification
pip install -r requirements.txt
```

### 4. **Run Models Locally**
Run the `main.py` script for the desired model:
```bash
python src/main.py
```

---

## **Dockerization**

### 1. **Build the Docker Image**
Navigate to the model folder (e.g., `bitcoin-model-classification`) and build the Docker image:
```bash
cd bitcoin-model-classification
docker build -t bitcoin-classification .
```

### 2. **Run the Docker Container**
Run the Docker container:
```bash
docker run -it --rm bitcoin-classification
```

---

## **Deployment**

### Deploy Models to the Cloud
Each model is designed to be deployed as a **Cloud Function** or another serverless environment. Refer to the **README.md** inside each model folder for specific deployment instructions.

---

## **Contributing**
If you'd like to contribute:
1. Fork the repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push:
   ```bash
   git commit -m "Add feature"
   git push origin feature-name
   ```
4. Open a pull request.

---

## **License**
This project is licensed under the [MIT License](LICENSE).

---

## **Acknowledgements**
- **Libraries Used**: `pandas`, `yfinance`, `scikit-learn`, `torch`, etc.
- Inspired by Bitcoin enthusiasts and data scientists!
