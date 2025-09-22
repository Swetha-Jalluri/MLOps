# 🍷 Wine Quality Prediction API - FastAPI_LabsNew

This project demonstrates how to build and deploy a machine learning model as a REST API using FastAPI. It uses Random Forest to predict wine quality based on physicochemical properties from the WineQT dataset.

## 📋 Overview

- **Model**: Random Forest Classifier
- **Dataset**: WineQT Dataset (1,143 samples, 11 features)
- **Framework**: FastAPI with uvicorn
- **Target**: Wine Quality Prediction (3-9 scale)
- **Features**: REST API, Auto-documentation, Health checks

## 🚀 What Makes This Project Unique

1. **Real Dataset**: Uses actual WineQT.csv dataset (not synthetic data)
2. **Production-Ready API**: Clean, simple structure matching FastAPI_Labs
3. **Proper Data Validation**: Pydantic models for input validation
4. **Feature Scaling**: Handles different scales of wine properties
5. **Error Handling**: Robust validation and error messages
6. **Same Structure**: Identical to FastAPI_Labs but with WineQT dataset

## 📁 Project Structure

```
FastAPI_LabsNew/
├── model/                      # Trained models
│   ├── wine_model.pkl         # Random Forest model
│   └── wine_scaler.pkl        # Feature scaler
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data.py                # Data loading utilities
│   ├── main.py                # FastAPI application
│   ├── predict.py             # Prediction logic
│   └── train.py               # Model training script
├── WineQT.csv                 # WineQT dataset
├── requirements.txt           # Dependencies
└── README.md
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup Steps

1. **Navigate to the project directory**
```bash
cd Labs/API_Labs/FastAPI_LabsNew
```

2. **Create virtual environment (optional)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

### 1. Train the Model

```bash
cd src
python train.py
```

This will:
- Load the WineQT.csv dataset
- Train the Random Forest model
- Save the trained model and scaler

### 2. Start the API Server

```bash
cd src
uvicorn main:app --host 127.0.0.1 --port 8000
```

The API will be available at: `http://127.0.0.1:8000`

## 📊 API Endpoints

### 🏠 Root (`/`)
Health check endpoint
```json
{"status": "healthy"}
```

### 🔮 Predict (`/predict`) [POST]
Predict wine quality

**Request Body:**
```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

**Response:**
```json
{
  "response": 5
}
```

### 📚 Documentation (`/docs`)
Interactive API documentation (Swagger UI)

## 🧪 Testing the API

### Using curl
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "fixed_acidity": 7.4,
       "volatile_acidity": 0.7,
       "citric_acid": 0.0,
       "residual_sugar": 1.9,
       "chlorides": 0.076,
       "free_sulfur_dioxide": 11.0,
       "total_sulfur_dioxide": 34.0,
       "density": 0.9978,
       "pH": 3.51,
       "sulphates": 0.56,
       "alcohol": 9.4
     }'
```

### Using Python
```python
import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

response = requests.post(url, json=data)
print(response.json())
```

## 🍷 Wine Quality Scale

| Score | Description |
|-------|-------------|
| 3-4   | Poor quality |
| 5     | Average quality |
| 6-7   | Good quality |
| 8-9   | Excellent quality |

## 📊 Dataset Information

**WineQT Dataset Features:**
- `fixed_acidity`: Fixed acidity (g/dm³)
- `volatile_acidity`: Volatile acidity (g/dm³)
- `citric_acid`: Citric acid (g/dm³)
- `residual_sugar`: Residual sugar (g/dm³)
- `chlorides`: Chlorides (g/dm³)
- `free_sulfur_dioxide`: Free sulfur dioxide (mg/dm³)
- `total_sulfur_dioxide`: Total sulfur dioxide (mg/dm³)
- `density`: Density (g/cm³)
- `pH`: pH level
- `sulphates`: Sulphates (g/dm³)
- `alcohol`: Alcohol content (% by volume)

## 🛠️ Technologies Used

- **FastAPI**: Modern web framework for building APIs
- **Random Forest**: Machine learning algorithm
- **Scikit-learn**: Machine learning utilities
- **Pandas/NumPy**: Data manipulation
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

## 🔄 Comparison with FastAPI_Labs

| Aspect | FastAPI_Labs | FastAPI_LabsNew |
|--------|--------------|-----------------|
| Dataset | Iris (4 features) | WineQT (11 features) |
| Model | Decision Tree | Random Forest |
| Target | Iris species | Wine quality |
| Structure | ✅ Identical | ✅ Identical |
| Endpoints | ✅ Same | ✅ Same |
| Response | ✅ Same format | ✅ Same format |

## 📝 Assignment Submission

This project is ready for submission with:
- ✅ Correct WineQT dataset implementation
- ✅ Clean, professional code structure
- ✅ Working API endpoints
- ✅ Proper documentation
- ✅ Same structure as FastAPI_Labs
- ✅ All requirements met

## 🙏 Credits

- Original Lab Structure: Based on FastAPI_Labs
- Dataset: WineQT Dataset
- Model: Random Forest implementation

## 📝 License

This project is for educational purposes.