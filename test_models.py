import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import onnxruntime as ort
from datetime import datetime

# CONFIGURATION
MODEL_PATH = 'models_pipeline'
TEST_OUTPUT_PATH = 'test'

# Model files
PIPELINE_PKL = os.path.join(MODEL_PATH, 'fraud_detection_pipeline.pkl')
XGBOOST_JSON = os.path.join(MODEL_PATH, 'xgboost_model.json')
ONNX_MODEL = os.path.join(MODEL_PATH, 'fraud_detection_xgboost.onnx')
SCALER_PATH = os.path.join(MODEL_PATH, 'scaler.pkl')

# Create test output directory
os.makedirs(TEST_OUTPUT_PATH, exist_ok=True)

# TEST DATA - 5 SAMPLES (Mix of Normal and Fraud)
test_samples = [
    # Sample 1: Normal transaction - Low amount, regular merchant
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
        "age": 57.0,
        "description": "Normal - Low amount grocery"
    },
    # Sample 2: Suspicious - High amount, unusual hour
    {
        "cc_num": 3573030041201292.0,
        "category": 5.0,
        "amt": 850.50,
        "gender": 0.0,
        "city": 16.0,
        "state": 44.0,
        "zip": 84002.0,
        "lat": 40.3207,
        "long": -110.436,
        "city_pop": 302.0,
        "merch_lat": 39.450498,
        "merch_long": -109.960431,
        "hour": 3.0,
        "dayofweek": 2.0,
        "month": 6.0,
        "age": 35.0,
        "description": "Suspicious - High amount at 3 AM"
    },
    # Sample 3: Normal transaction - Moderate amount
    {
        "cc_num": 3598215285024754.0,
        "category": 5.0,
        "amt": 41.28,
        "gender": 0.0,
        "city": 64.0,
        "state": 34.0,
        "zip": 11710.0,
        "lat": 40.6729,
        "long": -73.5365,
        "city_pop": 34496.0,
        "merch_lat": 40.49581,
        "merch_long": -74.196111,
        "hour": 14.0,
        "dayofweek": 3.0,
        "month": 6.0,
        "age": 55.0,
        "description": "Normal - Moderate shopping"
    },
    # Sample 4: Suspicious - Very high amount, distant merchant
    {
        "cc_num": 3591919803438423.0,
        "category": 9.0,
        "amt": 1250.99,
        "gender": 1.0,
        "city": 803.0,
        "state": 9.0,
        "zip": 32780.0,
        "lat": 28.5697,
        "long": -80.8191,
        "city_pop": 54767.0,
        "merch_lat": 35.812398,
        "merch_long": -85.883061,
        "hour": 23.0,
        "dayofweek": 0.0,
        "month": 12.0,
        "age": 38.0,
        "description": "Suspicious - High amount, distant location, late night"
    },
    # Sample 5: Normal - Small transaction, close merchant
    {
        "cc_num": 3526826139003047.0,
        "category": 13.0,
        "amt": 3.19,
        "gender": 1.0,
        "city": 261.0,
        "state": 22.0,
        "zip": 49632.0,
        "lat": 44.2529,
        "long": -85.017,
        "city_pop": 1126.0,
        "merch_lat": 44.959148,
        "merch_long": -85.884734,
        "hour": 10.0,
        "dayofweek": 4.0,
        "month": 3.0,
        "age": 70.0,
        "description": "Normal - Low amount, morning transaction"
    }
]

# Convert to DataFrame (remove description for model input)
test_df = pd.DataFrame([{k: v for k, v in sample.items() if k != 'description'} 
                        for sample in test_samples])


print("FRAUD DETECTION MODEL TESTING")
print(f"\nTest Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Number of test samples: {len(test_samples)}")
print(f"Output directory: {TEST_OUTPUT_PATH}\n")

# METHOD 1: PICKLE PIPELINE (FULL PIPELINE)
print("METHOD 1: PICKLE PIPELINE (fraud_detection_pipeline.pkl)")

results_pkl = []
try:
    # Load pipeline
    pipeline = joblib.load(PIPELINE_PKL)
    print("✓ Pipeline loaded successfully")
    
    # Predict
    predictions = pipeline.predict(test_df)
    probabilities = pipeline.predict_proba(test_df)[:, 1]
    
    print("\nResults:")
    print(f"{'ID':<4} {'Description':<45} {'Pred':<6} {'Prob':<10} {'Status'}")
    print("-"*80)
    
    for i, (sample, pred, prob) in enumerate(zip(test_samples, predictions, probabilities)):
        status = "FRAUD" if pred == 1 else "Normal"
        print(f"{i+1:<4} {sample['description']:<45} {pred:<6} {prob:<10.6f} {status}")
        
        results_pkl.append({
            "id": i + 1,
            "description": sample['description'],
            "input": {k: v for k, v in sample.items() if k != 'description'},
            "prediction": int(pred),
            "fraud_probability": float(prob),
            "is_fraud": bool(pred == 1)
        })
    
    # Save results
    pkl_results_path = os.path.join(TEST_OUTPUT_PATH, 'results_pickle_pipeline.json')
    with open(pkl_results_path, 'w') as f:
        json.dump(results_pkl, f, indent=2)
    print(f"\nResults saved to: {pkl_results_path}")
    
except Exception as e:
    print(f"✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()

# METHOD 2: XGBOOST NATIVE (MANUAL PREPROCESSING)
print("METHOD 2: XGBOOST NATIVE (xgboost_model.json)")

results_xgb = []
try:
    # Load scaler and model
    scaler = joblib.load(SCALER_PATH)
    xgb_model = XGBClassifier()
    xgb_model.load_model(XGBOOST_JSON)
    print("✓ Scaler and XGBoost model loaded successfully")
    
    # Manual preprocessing
    test_scaled = scaler.transform(test_df)
    
    # Predict
    predictions = xgb_model.predict(test_scaled)
    probabilities = xgb_model.predict_proba(test_scaled)[:, 1]
    
    print("\nResults:")
    print(f"{'ID':<4} {'Description':<45} {'Pred':<6} {'Prob':<10} {'Status'}")
    print("-"*80)
    
    for i, (sample, pred, prob) in enumerate(zip(test_samples, predictions, probabilities)):
        status = "FRAUD" if pred == 1 else "Normal"
        print(f"{i+1:<4} {sample['description']:<45} {pred:<6} {prob:<10.6f} {status}")
        
        results_xgb.append({
            "id": i + 1,
            "description": sample['description'],
            "input": {k: v for k, v in sample.items() if k != 'description'},
            "prediction": int(pred),
            "fraud_probability": float(prob),
            "is_fraud": bool(pred == 1)
        })
    
    # Verify consistency with pickle
    if results_pkl:
        pkl_preds = [r['prediction'] for r in results_pkl]
        xgb_preds = [r['prediction'] for r in results_xgb]
        pkl_probs = [r['fraud_probability'] for r in results_pkl]
        xgb_probs = [r['fraud_probability'] for r in results_xgb]
        
        pred_match = np.array_equal(pkl_preds, xgb_preds)
        prob_match = np.allclose(pkl_probs, xgb_probs, rtol=1e-5)
        
        print(f"\nConsistency Check with Pickle:")
        print(f"  Predictions match: {'✓ Yes' if pred_match else '✗ No'}")
        print(f"  Probabilities match: {'✓ Yes' if prob_match else '✗ No'}")
    
    # Save results
    xgb_results_path = os.path.join(TEST_OUTPUT_PATH, 'results_xgboost_native.json')
    with open(xgb_results_path, 'w') as f:
        json.dump(results_xgb, f, indent=2)
    print(f"\n✓ Results saved to: {xgb_results_path}")
    
except Exception as e:
    print(f"✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()

# METHOD 3: ONNX INFERENCE
print("METHOD 3: ONNX RUNTIME (fraud_detection_xgboost.onnx)")

results_onnx = []
try:
    # Load ONNX session
    sess = ort.InferenceSession(ONNX_MODEL)
    
    # Get input/output info
    input_name = sess.get_inputs()[0].name
    output_names = [output.name for output in sess.get_outputs()]
    
    print(f"✓ ONNX model loaded successfully")
    print(f"  Input name: {input_name}")
    print(f"  Output names: {output_names}")
    
    # Prepare input (must be scaled and float32)
    test_scaled = scaler.transform(test_df).astype(np.float32)
    
    # Run inference
    onnx_outputs = sess.run(output_names, {input_name: test_scaled})
    
    # Extract predictions and probabilities
    predictions = onnx_outputs[0].flatten()
    
    if len(onnx_outputs) > 1:
        probs_array = onnx_outputs[1]
        if probs_array.shape[1] == 2:
            probabilities = probs_array[:, 1]
        else:
            probabilities = probs_array.flatten()
    else:
        probabilities = predictions
    
    print("\nResults:")
    print(f"{'ID':<4} {'Description':<45} {'Pred':<6} {'Prob':<10} {'Status'}")
    print("-"*80)
    
    for i, (sample, pred, prob) in enumerate(zip(test_samples, predictions, probabilities)):
        pred_int = int(pred)
        status = "FRAUD" if pred_int == 1 else "Normal"
        print(f"{i+1:<4} {sample['description']:<45} {pred_int:<6} {prob:<10.6f} {status}")
        
        results_onnx.append({
            "id": i + 1,
            "description": sample['description'],
            "input": {k: v for k, v in sample.items() if k != 'description'},
            "prediction": pred_int,
            "fraud_probability": float(prob),
            "is_fraud": bool(pred_int == 1)
        })
    
    # Verify consistency with pickle
    if results_pkl:
        pkl_preds = [r['prediction'] for r in results_pkl]
        onnx_preds = [r['prediction'] for r in results_onnx]
        pkl_probs = [r['fraud_probability'] for r in results_pkl]
        onnx_probs = [r['fraud_probability'] for r in results_onnx]
        
        pred_match = np.allclose(pkl_preds, onnx_preds, rtol=1e-5)
        prob_match = np.allclose(pkl_probs, onnx_probs, rtol=1e-4)
        
        print(f"\nConsistency Check with Pickle:")
        print(f"  Predictions match: {'✓ Yes' if pred_match else '✗ No'}")
        print(f"  Probabilities match: {'✓ Yes' if prob_match else '✗ No'}")
    
    # Save results
    onnx_results_path = os.path.join(TEST_OUTPUT_PATH, 'results_onnx.json')
    with open(onnx_results_path, 'w') as f:
        json.dump(results_onnx, f, indent=2)
    print(f"\nResults saved to: {onnx_results_path}")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

# SUMMARY
print("TESTING SUMMARY")

if results_pkl and results_xgb and results_onnx:
    print("\n All three methods executed successfully!")
    
    # Count fraud detections
    fraud_count_pkl = sum(1 for r in results_pkl if r['is_fraud'])
    fraud_count_xgb = sum(1 for r in results_xgb if r['is_fraud'])
    fraud_count_onnx = sum(1 for r in results_onnx if r['is_fraud'])
    
    print(f"\nFraud Detection Count:")
    print(f"  Pickle Pipeline: {fraud_count_pkl}/{len(results_pkl)} transactions")
    print(f"  XGBoost Native:  {fraud_count_xgb}/{len(results_xgb)} transactions")
    print(f"  ONNX Runtime:    {fraud_count_onnx}/{len(results_onnx)} transactions")
    
    # Create consolidated report
    consolidated = {
        "test_info": {
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_samples": len(test_samples),
            "models_tested": ["pickle_pipeline", "xgboost_native", "onnx"]
        },
        "results": {
            "pickle_pipeline": results_pkl,
            "xgboost_native": results_xgb,
            "onnx_runtime": results_onnx
        }
    }
    
    consolidated_path = os.path.join(TEST_OUTPUT_PATH, 'consolidated_results.json')
    with open(consolidated_path, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    print(f"\n Consolidated results saved to: {consolidated_path}")
    
    # Save test inputs only (for API testing)
    test_inputs = [{
        "id": i + 1,
        "description": sample['description'],
        "input": {k: v for k, v in sample.items() if k != 'description'}
    } for i, sample in enumerate(test_samples)]
    
    inputs_path = os.path.join(TEST_OUTPUT_PATH, 'test_inputs.json')
    with open(inputs_path, 'w') as f:
        json.dump(test_inputs, f, indent=2)
    print(f"Test inputs saved to: {inputs_path}")

print("TESTING COMPLETE!")
print(f"\nAll results saved in: {TEST_OUTPUT_PATH}/")
print("Files created:")
print("  - results_pickle_pipeline.json")
print("  - results_xgboost_native.json")
print("  - results_onnx.json")
print("  - consolidated_results.json")
print("  - test_inputs.json")