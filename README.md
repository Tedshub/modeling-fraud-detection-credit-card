# Credit Card Fraud Detection Model

A machine learning project for detecting fraudulent credit card transactions using XGBoost classifier. This project provides multiple model export formats (Pickle, XGBoost Native JSON, and ONNX) for flexible deployment options.

## Project Description

This project implements a binary classification model to identify fraudulent credit card transactions. The model is trained on a comprehensive dataset containing transaction details such as amount, location, time, merchant information, and cardholder demographics. The XGBoost algorithm is used for its superior performance in handling imbalanced datasets and providing high accuracy in fraud detection.

### Key Features

- **Multiple Model Formats**: Supports Pickle (.pkl), XGBoost Native (.json), and ONNX (.onnx) formats
- **High Performance**: Achieves 99.84% accuracy with ROC-AUC of 0.9944
- **Production-Ready**: Includes preprocessing pipeline and example test cases
- **Flexible Deployment**: Choose the best format for your deployment environment

## Model Performance

The trained model achieves the following metrics on the test dataset:

```
PERFORMANCE METRICS
===========================================
Accuracy:  0.9984
Precision: 0.7964
Recall:    0.7916
F1-Score:  0.7940
ROC-AUC:   0.9944
===========================================

Classification Report:
              precision    recall  f1-score   support

   Non-Fraud       1.00      1.00      1.00    553574
       Fraud       0.80      0.79      0.79      2145

    accuracy                           1.00    555719
   macro avg       0.90      0.90      0.90    555719
weighted avg       1.00      1.00      1.00    555719

Confusion Matrix:
[[553140    434]
 [   447   1698]]
```

### Key Metrics Explanation

- **Accuracy (99.84%)**: Overall correctness of predictions
- **Precision (79.64%)**: Of all predicted frauds, 79.64% were actually fraudulent
- **Recall (79.16%)**: Of all actual frauds, the model detected 79.16%
- **F1-Score (79.40%)**: Harmonic mean of precision and recall
- **ROC-AUC (99.44%)**: Excellent discrimination ability between classes

## Project Structure

```
modeling-fraud-detection-credit-card/
│
├── dataset-fraud/                  # Dataset directory (create manually)
│   ├── fraudTrain.csv             # Training dataset
│   └── fraudTest.csv              # Testing dataset
│
├── models_pipeline/               # Trained models directory
│   ├── fraud_detection_pipeline.pkl       # Pickle pipeline (full)
│   ├── xgboost_model.json                # XGBoost native format
│   ├── fraud_detection_xgboost.onnx      # ONNX format
│   ├── scaler.pkl                        # StandardScaler object
│   ├── example_input.json                # Single input example
│   ├── example_with_output.json          # Input with expected output
│   └── test_cases.json                   # 10 test cases
│
├── test/                          # Test results directory
│   ├── results_pickle_pipeline.json
│   ├── results_xgboost_native.json
│   ├── results_onnx.json
│   ├── consolidated_results.json
│   └── test_inputs.json
│
├── train_model.py                 # Main training script
├── test_models.py                 # Model testing script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation & Setup

### Step 1: Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On Linux/Mac:
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare Dataset

1. Create dataset directory:
```bash
mkdir dataset-fraud
```

2. Download the dataset from Kaggle:
   - Link: https://www.kaggle.com/datasets/kartik2112/fraud-detection
   - Dataset contains two CSV files:
     - `fraudTrain.csv` (training data)
     - `fraudTest.csv` (testing data)

3. Extract and move the CSV files to `dataset-fraud/` directory

### Step 4: Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train the XGBoost model
- Export models in three formats (Pickle, JSON, ONNX)
- Generate evaluation metrics
- Create example test cases

### Step 5: Test the Models

```bash
python test_models.py
```

This will test all three model formats with 5 sample transactions and generate detailed results in the `test/` directory.

## Model Formats and Deployment Options

### 1. Pickle Pipeline (.pkl)

**File**: `fraud_detection_pipeline.pkl`

**Advantages**:
- Easiest to use - single file contains preprocessing + model
- No manual preprocessing required
- Best for Python-based deployments
- Fastest development and testing
- Maintains exact sklearn pipeline structure

**Use Cases**:
- Flask/FastAPI Python web services
- Jupyter notebook analysis
- Python-based batch processing
- Quick prototyping and testing

**Usage Example**:
```python
import joblib
pipeline = joblib.load('models_pipeline/fraud_detection_pipeline.pkl')
prediction = pipeline.predict(input_data)
probability = pipeline.predict_proba(input_data)[:, 1]
```

### 2. XGBoost Native JSON (.json)

**File**: `xgboost_model.json`

**Advantages**:
- Language-agnostic format
- Smaller file size than pickle
- Can be loaded in any language with XGBoost support (Python, R, Java, C++)
- Better version control (human-readable JSON)
- No Python version dependency issues
- More secure than pickle (no arbitrary code execution)

**Use Cases**:
- Cross-language deployments
- Java/Scala backend systems
- R-based analytics platforms
- Microservices architecture
- Mobile applications (via XGBoost mobile libraries)

**Usage Example**:
```python
from xgboost import XGBClassifier
import joblib

scaler = joblib.load('models_pipeline/scaler.pkl')
model = XGBClassifier()
model.load_model('models_pipeline/xgboost_model.json')

# Manual preprocessing required
scaled_data = scaler.transform(input_data)
prediction = model.predict(scaled_data)
```

### 3. ONNX Format (.onnx)

**File**: `fraud_detection_xgboost.onnx`

**Advantages**:
- Universal format supported by many frameworks
- Optimized for inference performance
- Hardware acceleration support (GPU, Neural Engine)
- Compatible with edge devices and IoT
- Cloud platform native support (Azure ML, AWS SageMaker)
- Minimal runtime dependencies
- Best inference speed

**Use Cases**:
- Edge computing and IoT devices
- Mobile applications (iOS, Android)
- Cloud ML platforms (Azure, AWS, GCP)
- High-performance production systems
- GPU-accelerated inference
- Embedded systems

**Usage Example**:
```python
import onnxruntime as ort
import joblib

scaler = joblib.load('models_pipeline/scaler.pkl')
session = ort.InferenceSession('models_pipeline/fraud_detection_xgboost.onnx')

# Prepare input
scaled_data = scaler.transform(input_data).astype(np.float32)
input_name = session.get_inputs()[0].name

# Run inference
result = session.run(None, {input_name: scaled_data})
predictions = result[0]
```

## Input Format

The model expects 16 features in the following format:

```json
{
  "cc_num": 2291163933867244.0,
  "category": 10.0,
  "amt": 2.86,
  "gender": 1.0,
  "city": 168.0,
  "state": 40.0,
  "zip": 29209.0,
  "lat": 33.9659,
  "long": -80.9355,
  "city_pop": 333497.0,
  "merch_lat": 33.986391,
  "merch_long": -81.200714,
  "hour": 12.0,
  "dayofweek": 6.0,
  "month": 6.0,
  "age": 57.0
}
```

### Feature Descriptions

- `cc_num`: Credit card number (encoded)
- `category`: Merchant category code (encoded)
- `amt`: Transaction amount
- `gender`: Gender (0=Female, 1=Male)
- `city`: City code (encoded)
- `state`: State code (encoded)
- `zip`: ZIP code
- `lat`: Cardholder latitude
- `long`: Cardholder longitude
- `city_pop`: City population
- `merch_lat`: Merchant latitude
- `merch_long`: Merchant longitude
- `hour`: Transaction hour (0-23)
- `dayofweek`: Day of week (0=Monday, 6=Sunday)
- `month`: Month (1-12)
- `age`: Cardholder age

## Output Format

The model returns predictions in the following format:

```json
{
  "prediction": 0,
  "fraud_probability": 1.3480442930813297e-06,
  "is_fraud": false
}
```

- `prediction`: Binary prediction (0=Non-Fraud, 1=Fraud)
- `fraud_probability`: Probability of fraud (0.0 to 1.0)
- `is_fraud`: Boolean flag for fraud status

## Test Cases

The repository includes ready-to-use test cases:

### Example Files

1. **example_input.json**: Single transaction input
2. **example_with_output.json**: Input with expected output
3. **test_cases.json**: 10 diverse test cases for validation

These files are located in `models_pipeline/` directory and can be used for:
- API testing
- Integration testing
- Model validation
- Performance benchmarking

## Model Comparison Summary

| Aspect | Pickle (.pkl) | XGBoost JSON (.json) | ONNX (.onnx) |
|--------|---------------|---------------------|--------------|
| Ease of Use | Excellent | Good | Good |
| File Size | Large | Small | Medium |
| Portability | Python only | Multi-language | Universal |
| Performance | Good | Good | Excellent |
| Security | Low | High | High |
| Preprocessing | Included | Manual required | Manual required |
| Best For | Python apps | Cross-platform | Production/Edge |

## Dependencies

Main libraries used in this project:

```
pandas
numpy
scikit-learn
xgboost
onnx
onnxruntime
skl2onnx
imbalanced-learn
joblib
```

See `requirements.txt` for complete list with versions.

## Model Training Details

- **Algorithm**: XGBoost Classifier
- **Training Samples**: 1,296,675 transactions
- **Test Samples**: 555,719 transactions
- **Class Distribution**: Highly imbalanced (0.35% fraud)
- **Handling Imbalance**: Class weighting with scale_pos_weight
- **Feature Scaling**: StandardScaler normalization
- **Cross-Validation**: Used for hyperparameter tuning

## License

This project is available for educational and research purposes.

## Dataset Source

Dataset: Fraud Detection
Source: Kaggle - https://www.kaggle.com/datasets/kartik2112/fraud-detection
Creator: Kartik Shenoy

## Contributing

Feel free to open issues or submit pull requests for improvements.

## Author

**Created by**: [Tedshub](https://github.com/Tedshub)

This repository was initially developed and maintained by Tedshub.

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This model is for educational purposes. For production use in financial systems, additional validation, monitoring, and compliance measures are required.
