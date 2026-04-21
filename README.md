# IoT Equipment ML Maintenance Predictor

This repository contains a machine learning pipeline that trains various scikit-learn models on IoT manufacturing data and exports the best pipeline. It also includes an interface application built with Streamlit for quick inference and testing.

## Files
- `iot_based_ml_monitoring.py`: Script to train and evaluate ML classifiers on IoT sensor data to determine maintenance requirements. The script automatically isolates the best performing model pipeline and saves it to `best_model.pkl`.
- `smart_manufacturing_data.csv`: The raw dataset consisting of real-time machine readouts, anomalies, and maintenance flags.
- `app.py`: A Streamlit web application dashboard that loads the pickle model and allows user input to determine if a machine requires maintenance interactively.
- `best_model.pkl`: The serialized `AdaBoostClassifier` model with its bundled `StandardScaler` standardizer.

## Setup and Installation

### 1. Create a Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Web Application
Run the following command inside an active virtual environment to start the Streamlit web dashboard:
```bash
streamlit run app.py
```
The browser app will boot up on [http://localhost:8501](http://localhost:8501)!
