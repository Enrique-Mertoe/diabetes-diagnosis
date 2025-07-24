#!/usr/bin/env python3
"""
Diabetes Risk Prediction AI Solution - Demo Script
Assignment 2: Solution Implementation

Author: [Your Name]
Date: 2024
Purpose: Demonstrate diabetes risk prediction using machine learning

This script provides a simple command-line interface to test the 
diabetes prediction model without requiring a web interface.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class DiabetesPredictor:
    """
    AI-powered diabetes risk prediction system using Random Forest classification.
    
    This class handles data preprocessing, model training, and prediction for 
    diabetes risk assessment based on medical indicators from the PIMA dataset.
    """
    
    def __init__(self, model_path=None):
        """Initialize the predictor with optional pre-trained model."""
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No pre-trained model found. Training new model...")
            self.train_model()
    
    def preprocess_data(self, df):
        """
        Clean and preprocess the diabetes dataset.
        
        Args:
            df (pandas.DataFrame): Raw dataset
            
        Returns:
            pandas.DataFrame: Cleaned dataset with zero values replaced
        """
        print("Preprocessing data...")
        df_clean = df.copy()
        
        # Replace biologically impossible zero values with median
        zero_replacement_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for col in zero_replacement_columns:
            if col in df_clean.columns:
                # Calculate median excluding zeros
                median_val = df_clean[df_clean[col] != 0][col].median()
                zero_count = (df_clean[col] == 0).sum()
                
                if zero_count > 0:
                    df_clean[col] = df_clean[col].replace(0, median_val)
                    print(f"  - Replaced {zero_count} zero values in {col} with median {median_val:.1f}")
        
        return df_clean
    
    def train_model(self, data_path='diabetes.csv'):
        """
        Train the diabetes prediction model.
        
        Args:
            data_path (str): Path to the diabetes dataset CSV file
        """
        print(f"\nTraining diabetes prediction model...")
        print("=" * 50)
        
        # Load and preprocess data
        try:
            df = pd.read_csv(data_path)
            print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        except FileNotFoundError:
            print(f"Error: Dataset file '{data_path}' not found!")
            print("Please ensure the PIMA diabetes dataset is available.")
            return
        
        df_clean = self.preprocess_data(df)
        
        # Prepare features and target
        X = df_clean[self.feature_columns]
        y = df_clean['Outcome']
        
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature scaling for some algorithms (though Random Forest doesn't require it)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        print("\nTraining Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Training Completed!")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        # Detailed evaluation
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        # Feature importance
        print("\nTop 5 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']:25s}: {row['importance']:.4f}")
        
        # Save model
        self.save_model('diabetes_model.joblib')
        print(f"\nModel saved to 'diabetes_model.joblib'")
    
    def predict(self, patient_data):
        """
        Predict diabetes risk for a patient.
        
        Args:
            patient_data (dict): Patient medical data with required features
            
        Returns:
            dict: Prediction results with probability and risk level
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Convert to DataFrame and ensure proper column order
        if isinstance(patient_data, dict):
            input_df = pd.DataFrame([patient_data])[self.feature_columns]
        else:
            input_df = pd.DataFrame(patient_data, columns=self.feature_columns)
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        probability = self.model.predict_proba(input_df)[0, 1]  # Probability of diabetes
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
            risk_description = "Low probability of diabetes"
        elif probability < 0.7:
            risk_level = "Moderate Risk"
            risk_description = "Moderate probability of diabetes - monitoring recommended"
        else:
            risk_level = "High Risk"
            risk_description = "High probability of diabetes - medical consultation advised"
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'description': risk_description,
            'confidence': float(max(self.model.predict_proba(input_df)[0]))
        }
    
    def save_model(self, filepath):
        """Save the trained model and scaler."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a pre-trained model and scaler."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {filepath}")

def get_patient_input():
    """
    Get patient information from user input.
    
    Returns:
        dict: Patient data with all required features
    """
    print("\n" + "="*60)
    print("DIABETES RISK ASSESSMENT - PATIENT DATA ENTRY")
    print("="*60)
    
    print("\nPlease enter the following patient information:")
    print("(Enter 0 if unknown - the system will use appropriate defaults)")
    
    patient_data = {}
    
    # Define input prompts with validation ranges
    inputs = [
        ('Pregnancies', 'Number of pregnancies', 0, 20),
        ('Glucose', 'Glucose level (mg/dL)', 0, 300),
        ('BloodPressure', 'Blood pressure (mmHg)', 0, 200),
        ('SkinThickness', 'Skin thickness (mm)', 0, 100),
        ('Insulin', 'Insulin level (μU/mL)', 0, 1000),
        ('BMI', 'Body Mass Index (BMI)', 0, 70),
        ('DiabetesPedigreeFunction', 'Diabetes pedigree function (0.0-3.0)', 0.0, 3.0),
        ('Age', 'Age (years)', 1, 120)
    ]
    
    for feature, prompt, min_val, max_val in inputs:
        while True:
            try:
                if feature == 'DiabetesPedigreeFunction':
                    value = float(input(f"{prompt}: "))
                else:
                    value = float(input(f"{prompt}: "))
                
                if min_val <= value <= max_val:
                    patient_data[feature] = value
                    break
                else:
                    print(f"Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")
    
    return patient_data

def demo_with_sample_patients(predictor):
    """Demonstrate the model with sample patients."""
    print("\n" + "="*60)
    print("DEMONSTRATION WITH SAMPLE PATIENTS")
    print("="*60)
    
    sample_patients = [
        {
            'name': 'High Risk Patient',
            'data': {
                'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72,
                'SkinThickness': 35, 'Insulin': 100, 'BMI': 33.6,
                'DiabetesPedigreeFunction': 0.627, 'Age': 50
            }
        },
        {
            'name': 'Low Risk Patient', 
            'data': {
                'Pregnancies': 1, 'Glucose': 85, 'BloodPressure': 66,
                'SkinThickness': 29, 'Insulin': 85, 'BMI': 26.6,
                'DiabetesPedigreeFunction': 0.351, 'Age': 31
            }
        },
        {
            'name': 'Moderate Risk Patient',
            'data': {
                'Pregnancies': 3, 'Glucose': 120, 'BloodPressure': 80,
                'SkinThickness': 25, 'Insulin': 100, 'BMI': 28.0,
                'DiabetesPedigreeFunction': 0.400, 'Age': 35
            }
        }
    ]
    
    for patient in sample_patients:
        print(f"\n{patient['name']}:")
        print("-" * 30)
        
        # Display patient data
        for feature, value in patient['data'].items():
            print(f"  {feature:25s}: {value}")
        
        # Make prediction
        result = predictor.predict(patient['data'])
        
        print(f"\nPrediction Results:")
        print(f"  Diabetes Risk: {'POSITIVE' if result['prediction'] == 1 else 'NEGATIVE'}")
        print(f"  Probability: {result['probability']:.3f} ({result['probability']*100:.1f}%)")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Description: {result['description']}")

def main():
    """Main function to run the diabetes prediction demo."""
    print("🩺 DIABETES RISK PREDICTION AI SYSTEM")
    print("=" * 60)
    print("Using machine learning to assess diabetes risk from medical indicators")
    print("Based on the PIMA Indian Diabetes Dataset")
    print("=" * 60)
    
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Demo with sample patients")
        print("2. Enter new patient data")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            demo_with_sample_patients(predictor)
            
        elif choice == '2':
            try:
                patient_data = get_patient_input()
                
                print(f"\nAnalyzing patient data...")
                result = predictor.predict(patient_data)
                
                print(f"\n" + "="*60)
                print("DIABETES RISK ASSESSMENT RESULTS")
                print("="*60)
                
                print(f"\n📊 PREDICTION SUMMARY:")
                print(f"  Diabetes Risk: {'⚠️  POSITIVE' if result['prediction'] == 1 else '✅ NEGATIVE'}")
                print(f"  Probability: {result['probability']:.3f} ({result['probability']*100:.1f}%)")
                print(f"  Risk Level: {result['risk_level']}")
                print(f"  Assessment: {result['description']}")
                
                print(f"\n⚠️  MEDICAL DISCLAIMER:")
                print(f"  This AI prediction is for educational purposes only.")
                print(f"  Always consult with healthcare professionals for medical advice.")
                
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                
        elif choice == '3':
            print("\nThank you for using the Diabetes Risk Prediction System!")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the diabetes.csv dataset is available in the current directory.")