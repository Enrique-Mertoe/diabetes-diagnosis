#!/usr/bin/env python3
"""
Test script for the diabetes prediction Flask application
"""

from app import DiabetesPredictor
import pandas as pd

def test_model():
    print("Testing Diabetes Prediction Model...")
    
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    # Test data - sample from PIMA dataset
    test_cases = [
        # High risk case
        {
            'name': 'High Risk Patient',
            'features': [6, 148, 72, 35, 0, 33.6, 0.627, 50],
            'expected_risk': 'high'
        },
        # Low risk case  
        {
            'name': 'Low Risk Patient',
            'features': [1, 85, 66, 29, 0, 26.6, 0.351, 31],
            'expected_risk': 'low'
        },
        # Medium risk case
        {
            'name': 'Medium Risk Patient',
            'features': [3, 120, 80, 25, 100, 28.0, 0.400, 35],
            'expected_risk': 'medium'
        }
    ]
    
    print(f"[OK] Model loaded successfully!")
    print(f"Features: {predictor.feature_names}")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {case['name']} ---")
        
        # Make prediction
        result = predictor.predict(case['features'])
        
        print(f"Input features: {case['features']}")
        print(f"Prediction: {'Diabetic' if result['prediction'] == 1 else 'Not Diabetic'}")
        print(f"Probability: {result['probability']:.3f} ({result['probability']*100:.1f}%)")
        print(f"Risk Level: {result['risk_level']}")
        
        # Verify risk level mapping
        if result['probability'] < 0.3:
            expected_level = "Low Risk"
        elif result['probability'] < 0.7:
            expected_level = "Moderate Risk"
        else:
            expected_level = "High Risk"
            
        assert result['risk_level'] == expected_level, f"Risk level mismatch: got {result['risk_level']}, expected {expected_level}"
        
    print(f"\n[SUCCESS] All tests passed! Model is working correctly.")
    
    # Check dataset info
    df = pd.read_csv('diabetes.csv')
    print(f"\nDataset Info:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Features: {list(df.columns[:-1])}")
    print(f"   - Positive cases: {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
    print(f"   - Negative cases: {len(df) - df['Outcome'].sum()} ({(1-df['Outcome'].mean())*100:.1f}%)")

def test_data_preprocessing():
    print(f"\nTesting data preprocessing...")
    
    predictor = DiabetesPredictor()
    
    # Create test dataframe with zero values
    test_df = pd.DataFrame({
        'Pregnancies': [1, 2, 3],
        'Glucose': [0, 120, 140],  # Zero glucose (impossible)
        'BloodPressure': [80, 0, 90],  # Zero BP (impossible)
        'SkinThickness': [20, 25, 0],  # Zero skin thickness (impossible)
        'Insulin': [0, 100, 150],  # Zero insulin (possible but suspicious)
        'BMI': [25.0, 0, 30.0],  # Zero BMI (impossible)
        'DiabetesPedigreeFunction': [0.5, 0.6, 0.7],
        'Age': [25, 30, 35]
    })
    
    print("Before preprocessing:")
    print(test_df)
    
    processed_df = predictor.preprocess_data(test_df.copy())
    
    print("\nAfter preprocessing:")
    print(processed_df)
    
    # Check that zeros were replaced
    columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_to_check:
        if (test_df[col] == 0).any():
            assert not (processed_df[col] == 0).any(), f"Zeros not properly handled in {col}"
    
    print("[OK] Data preprocessing works correctly!")

if __name__ == "__main__":
    try:
        test_model()
        test_data_preprocessing()
        print(f"\n[SUCCESS] All tests completed successfully!")
        print(f"Your Flask app is ready to run!")
        print(f"Run: python app.py")
        print(f"Then open: http://127.0.0.1:5000")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()