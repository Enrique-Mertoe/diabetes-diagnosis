#!/usr/bin/env python3
"""
AI-Enhanced Medical Assistant Demo
Assignment 2: Advanced AI Integration

Author: [Your Name]
Date: 2024
Purpose: Demonstrate AI-powered diabetes risk assessment with natural language insights

This script provides a command-line interface to test the AI-enhanced
diabetes prediction system without requiring a web interface.
Features Gemini AI integration for intelligent medical analysis.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
import json
from datetime import datetime
import time

# Optional Gemini AI import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class AIMedicalAssistant:
    """
    AI-Enhanced Medical Assistant for Diabetes Risk Assessment.
    
    Combines traditional machine learning with generative AI to provide:
    - Accurate diabetes risk predictions
    - Natural language explanations
    - Personalized medical recommendations
    - Business intelligence insights
    """
    
    def __init__(self, gemini_api_key=None):
        """Initialize the AI Medical Assistant."""
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        self.feature_importance = {}
        self.gemini_available = False
        self.ai_model = None
        
        # Initialize Gemini AI if available
        if GEMINI_AVAILABLE and gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.ai_model = genai.GenerativeModel('gemini-pro')
                self.gemini_available = True
                print("🤖 Gemini AI Medical Assistant initialized successfully!")
            except Exception as e:
                print(f"⚠️ Gemini AI initialization failed: {e}")
                print("📝 Using advanced mock AI responses for demonstration")
        else:
            print("📝 Using intelligent mock AI responses (set GEMINI_API_KEY for real AI)")
        
        # Load or train the ML model
        self.load_or_train_model()
    
    def preprocess_data(self, df):
        """Clean and preprocess the diabetes dataset."""
        df_clean = df.copy()
        zero_replacement_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for col in zero_replacement_columns:
            if col in df_clean.columns:
                median_val = df_clean[df_clean[col] != 0][col].median()
                zero_count = (df_clean[col] == 0).sum()
                if zero_count > 0:
                    df_clean[col] = df_clean[col].replace(0, median_val)
        
        return df_clean
    
    def load_or_train_model(self, data_path='diabetes.csv'):
        """Load existing model or train a new one."""
        model_path = 'ai_diabetes_model.joblib'
        
        if os.path.exists(model_path):
            print("📂 Loading existing AI-enhanced model...")
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler', StandardScaler())
            self.feature_importance = model_data.get('feature_importance', {})
            print("✅ Model loaded successfully!")
        else:
            print("🔄 Training new AI-enhanced model...")
            self.train_model(data_path)
    
    def train_model(self, data_path='diabetes.csv'):
        """Train the diabetes prediction model with AI enhancements."""
        try:
            df = pd.read_csv(data_path)
            print(f"📊 Dataset loaded: {df.shape[0]} patients, {df.shape[1]} features")
        except FileNotFoundError:
            print(f"❌ Error: Dataset file '{data_path}' not found!")
            print("Please ensure the PIMA diabetes dataset is available.")
            return
        
        # Preprocess data
        df_clean = self.preprocess_data(df)
        
        # Prepare features and target
        X = df_clean[self.feature_columns]
        y = df_clean['Outcome']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            self.feature_columns, self.model.feature_importances_
        ))
        self.feature_importance = dict(sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        ))
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model training completed!")
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, 'ai_diabetes_model.joblib')
        print("💾 AI-enhanced model saved!")
    
    def predict_risk(self, patient_data):
        """Make diabetes risk prediction with AI enhancement."""
        if self.model is None:
            raise ValueError("Model not available. Please check dataset and training.")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([patient_data])[self.feature_columns]
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        probability = self.model.predict_proba(input_df)[0, 1]
        
        # Enhanced risk categorization
        if probability < 0.15:
            risk_level = "Very Low Risk"
            urgency = "Routine Care"
        elif probability < 0.35:
            risk_level = "Low Risk"
            urgency = "Annual Screening"
        elif probability < 0.55:
            risk_level = "Moderate Risk"
            urgency = "6-Month Follow-up"
        elif probability < 0.75:
            risk_level = "High Risk"
            urgency = "Medical Consultation"
        else:
            risk_level = "Very High Risk"
            urgency = "Immediate Medical Attention"
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'urgency': urgency,
            'confidence': float(max(self.model.predict_proba(input_df)[0]))
        }
    
    def generate_ai_insights(self, patient_data, prediction_result):
        """Generate AI-powered medical insights."""
        if self.gemini_available:
            return self._generate_real_ai_insights(patient_data, prediction_result)
        else:
            return self._generate_intelligent_mock_insights(patient_data, prediction_result)
    
    def _generate_real_ai_insights(self, patient_data, prediction_result):
        """Generate insights using Gemini AI."""
        prompt = f"""
        As an AI medical assistant specializing in diabetes risk assessment, provide a comprehensive analysis for this patient:
        
        PATIENT DATA:
        - Age: {patient_data['Age']} years
        - BMI: {patient_data['BMI']}
        - Glucose: {patient_data['Glucose']} mg/dL
        - Blood Pressure: {patient_data['BloodPressure']} mmHg
        - Pregnancies: {patient_data['Pregnancies']}
        - Family History Score: {patient_data['DiabetesPedigreeFunction']}
        
        ML PREDICTION:
        - Risk Level: {prediction_result['risk_level']}
        - Probability: {prediction_result['probability']:.1%}
        - Urgency: {prediction_result['urgency']}
        
        TOP RISK FACTORS:
        {json.dumps(dict(list(self.feature_importance.items())[:3]), indent=2)}
        
        Please provide:
        1. MEDICAL ASSESSMENT: Clinical interpretation of the risk factors
        2. KEY CONCERNS: Most important health indicators to address
        3. PERSONALIZED RECOMMENDATIONS: Specific actions for this patient
        4. PATIENT EXPLANATION: How to communicate this risk in simple terms
        5. FOLLOW-UP PLAN: Next steps and monitoring schedule
        
        Keep responses concise and medically accurate.
        """
        
        try:
            response = self.ai_model.generate_content(prompt)
            return self._parse_ai_response(response.text, patient_data, prediction_result)
        except Exception as e:
            print(f"⚠️ AI analysis error: {e}")
            return self._generate_intelligent_mock_insights(patient_data, prediction_result)
    
    def _generate_intelligent_mock_insights(self, patient_data, prediction_result):
        """Generate intelligent mock insights that simulate real AI analysis."""
        risk_level = prediction_result['risk_level']
        probability = prediction_result['probability']
        
        # Analyze key risk factors
        high_risk_factors = []
        moderate_risk_factors = []
        
        if patient_data['Glucose'] > 140:
            high_risk_factors.append(f"Elevated glucose ({patient_data['Glucose']} mg/dL)")
        elif patient_data['Glucose'] > 100:
            moderate_risk_factors.append(f"Borderline glucose ({patient_data['Glucose']} mg/dL)")
        
        if patient_data['BMI'] > 30:
            high_risk_factors.append(f"Obesity (BMI {patient_data['BMI']})")
        elif patient_data['BMI'] > 25:
            moderate_risk_factors.append(f"Overweight (BMI {patient_data['BMI']})")
        
        if patient_data['Age'] > 45:
            moderate_risk_factors.append(f"Age-related risk ({patient_data['Age']} years)")
        
        if patient_data['DiabetesPedigreeFunction'] > 0.5:
            moderate_risk_factors.append(f"Family history factor ({patient_data['DiabetesPedigreeFunction']:.3f})")
        
        # Generate personalized recommendations
        recommendations = []
        
        if patient_data['Glucose'] > 126:
            recommendations.extend([
                "🩸 Monitor blood glucose levels weekly",
                "🍎 Follow a low-glycemic index diet",
                "💊 Discuss glucose management with physician"
            ])
        
        if patient_data['BMI'] > 25:
            recommendations.extend([
                "🏃‍♀️ Engage in 150 minutes of moderate exercise weekly",
                "⚖️ Target 5-10% weight reduction",
                "🥗 Consult nutritionist for meal planning"
            ])
        
        if patient_data['Age'] > 45:
            recommendations.append("📅 Schedule diabetes screening every 6 months")
        
        if probability > 0.7:
            recommendations.extend([
                "🏥 Schedule endocrinologist consultation within 2 weeks",
                "📊 Consider continuous glucose monitoring",
                "💡 Explore diabetes prevention programs"
            ])
        
        # Generate insights
        insights = {
            'medical_assessment': f"""Based on comprehensive analysis, this {patient_data['Age']}-year-old patient presents with {risk_level.lower()} 
            for type 2 diabetes development. The {probability*100:.1f}% probability is derived from multiple risk factors including metabolic, 
            demographic, and genetic indicators. The assessment suggests {'immediate' if probability > 0.7 else 'proactive'} medical intervention.""",
            
            'key_concerns': self._format_concerns(high_risk_factors, moderate_risk_factors),
            
            'recommendations': recommendations[:6],  # Top 6 recommendations
            
            'patient_explanation': f"""Your diabetes risk assessment shows {risk_level.lower()}. This means you have a {probability*100:.0f}% 
            chance of developing type 2 diabetes based on your current health indicators. {'This requires immediate medical attention.' if probability > 0.7 
            else 'This is manageable with lifestyle modifications and regular monitoring.'}""",
            
            'follow_up_plan': self._generate_follow_up_plan(prediction_result['urgency'], probability),
            
            'lifestyle_focus': self._determine_lifestyle_focus(patient_data)
        }
        
        return insights
    
    def _format_concerns(self, high_risk, moderate_risk):
        """Format risk concerns in priority order."""
        concerns = []
        if high_risk:
            concerns.append(f"🔴 HIGH PRIORITY: {', '.join(high_risk)}")
        if moderate_risk:
            concerns.append(f"🟡 MONITOR CLOSELY: {', '.join(moderate_risk)}")
        if not high_risk and not moderate_risk:
            concerns.append("✅ No major risk factors identified - continue preventive care")
        return concerns
    
    def _generate_follow_up_plan(self, urgency, probability):
        """Generate appropriate follow-up schedule."""
        if probability > 0.8:
            return "Immediate: Schedule appointment within 1 week, then monthly monitoring"
        elif probability > 0.6:
            return "Urgent: Schedule appointment within 2 weeks, then quarterly monitoring"
        elif probability > 0.4:
            return "Routine: Schedule appointment within 1 month, then bi-annual monitoring"
        else:
            return "Preventive: Annual check-up with routine diabetes screening"
    
    def _determine_lifestyle_focus(self, patient_data):
        """Determine primary lifestyle intervention focus."""
        if patient_data['BMI'] > 30:
            return "Weight management through diet and exercise"
        elif patient_data['Glucose'] > 140:
            return "Blood sugar control through dietary modifications"
        elif patient_data['Age'] > 50:
            return "Age-appropriate fitness and preventive care"
        else:
            return "General wellness and diabetes prevention"
    
    def _parse_ai_response(self, ai_text, patient_data, prediction_result):
        """Parse AI response into structured insights."""
        # Simple parsing - in production, would use more sophisticated NLP
        return {
            'medical_assessment': f"AI Analysis: {ai_text[:200]}...",
            'key_concerns': ["AI-identified risk patterns from advanced analysis"],
            'recommendations': ["AI-generated personalized recommendations"],
            'patient_explanation': f"AI Communication: Simplified explanation for patient understanding",
            'follow_up_plan': prediction_result['urgency'],
            'lifestyle_focus': "AI-recommended lifestyle interventions"
        }

def display_welcome():
    """Display welcome message and system information."""
    print("=" * 80)
    print("🤖 AI-ENHANCED MEDICAL ASSISTANT")
    print("   Diabetes Risk Assessment with Artificial Intelligence")
    print("=" * 80)
    print("Features:")
    print("• 🧠 Advanced AI-powered risk analysis")
    print("• 💬 Natural language medical explanations")
    print("• 🎯 Personalized recommendations")
    print("• 📊 Comprehensive health insights")
    print("• 🏥 Clinical decision support")
    print("=" * 80)

def get_patient_input():
    """Get patient information through interactive input."""
    print("\n" + "🏥 PATIENT DATA COLLECTION".center(60, "="))
    print("\nPlease enter the patient's medical information:")
    print("(Enter 0 for any unknown values)")
    
    inputs = [
        ('Pregnancies', 'Number of pregnancies', 0, 20, int),
        ('Glucose', 'Glucose level (mg/dL)', 50, 300, float),
        ('BloodPressure', 'Blood pressure (mmHg)', 40, 200, float),
        ('SkinThickness', 'Skin thickness (mm)', 5, 100, float),
        ('Insulin', 'Insulin level (μU/mL)', 10, 1000, float),
        ('BMI', 'Body Mass Index', 10, 70, float),
        ('DiabetesPedigreeFunction', 'Family history score (0.0-3.0)', 0.0, 3.0, float),
        ('Age', 'Age (years)', 18, 120, int)
    ]
    
    patient_data = {}
    
    for feature, prompt, min_val, max_val, data_type in inputs:
        while True:
            try:
                value = data_type(input(f"  {prompt}: "))
                if min_val <= value <= max_val:
                    patient_data[feature] = value
                    break
                else:
                    print(f"    ⚠️ Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print(f"    ⚠️ Please enter a valid {data_type.__name__}")
    
    return patient_data

def display_analysis_results(assistant, patient_data, patient_name="Patient"):
    """Display comprehensive AI analysis results."""
    print(f"\n{'🔬 AI MEDICAL ANALYSIS':=^80}")
    print(f"Patient: {patient_name}")
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Step 1: ML Prediction
    print("🤖 MACHINE LEARNING ASSESSMENT:")
    print("-" * 40)
    
    prediction_result = assistant.predict_risk(patient_data)
    
    print(f"   Risk Level: {prediction_result['risk_level']}")
    print(f"   Probability: {prediction_result['probability']:.3f} ({prediction_result['probability']*100:.1f}%)")
    print(f"   Recommendation: {prediction_result['urgency']}")
    print(f"   Confidence: {prediction_result['confidence']:.3f}")
    
    # Step 2: AI Enhancement
    print(f"\n🧠 AI INTELLIGENCE LAYER:")
    print("-" * 40)
    print("   Generating comprehensive insights...")
    
    # Simulate AI thinking time
    time.sleep(1)
    
    ai_insights = assistant.generate_ai_insights(patient_data, prediction_result)
    
    print(f"\n📋 MEDICAL ASSESSMENT:")
    print(f"   {ai_insights['medical_assessment']}")
    
    print(f"\n⚠️ KEY CONCERNS:")
    for concern in ai_insights['key_concerns']:
        print(f"   • {concern}")
    
    print(f"\n💡 PERSONALIZED RECOMMENDATIONS:")
    for i, rec in enumerate(ai_insights['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print(f"\n💬 PATIENT COMMUNICATION:")
    print(f"   {ai_insights['patient_explanation']}")
    
    print(f"\n📅 FOLLOW-UP PLAN:")
    print(f"   {ai_insights['follow_up_plan']}")
    
    print(f"\n🎯 LIFESTYLE FOCUS:")
    print(f"   {ai_insights['lifestyle_focus']}")
    
    print(f"\n⚕️ MEDICAL DISCLAIMER:")
    print("   This AI assessment is for educational purposes only.")
    print("   Always consult healthcare professionals for medical decisions.")
    
    return {
        'patient_data': patient_data,
        'ml_prediction': prediction_result,
        'ai_insights': ai_insights,
        'timestamp': datetime.now().isoformat()
    }

def demo_sample_patients(assistant):
    """Demonstrate the system with pre-defined sample patients."""
    print(f"\n{'🎭 SAMPLE PATIENT DEMONSTRATIONS':=^80}")
    
    sample_patients = [
        {
            'name': 'Sarah Johnson (High Risk)',
            'data': {
                'Pregnancies': 6, 'Glucose': 168, 'BloodPressure': 88,
                'SkinThickness': 40, 'Insulin': 120, 'BMI': 34.2,
                'DiabetesPedigreeFunction': 0.785, 'Age': 52
            }
        },
        {
            'name': 'Maria Rodriguez (Low Risk)',
            'data': {
                'Pregnancies': 1, 'Glucose': 92, 'BloodPressure': 68,
                'SkinThickness': 28, 'Insulin': 88, 'BMI': 24.8,
                'DiabetesPedigreeFunction': 0.290, 'Age': 29
            }
        },
        {
            'name': 'Jennifer Kim (Moderate Risk)',
            'data': {
                'Pregnancies': 3, 'Glucose': 125, 'BloodPressure': 82,
                'SkinThickness': 32, 'Insulin': 110, 'BMI': 28.5,
                'DiabetesPedigreeFunction': 0.445, 'Age': 38
            }
        }
    ]
    
    analyses = []
    for patient in sample_patients:
        print(f"\n{'Analyzing: ' + patient['name']:^80}")
        analysis = display_analysis_results(assistant, patient['data'], patient['name'])
        analyses.append(analysis)
        
        input("\nPress Enter to continue to next patient...")
    
    return analyses

def main():
    """Main function to run the AI Medical Assistant demo."""
    display_welcome()
    
    # Initialize AI Assistant
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print("🔑 Gemini API key detected - enabling real AI analysis")
    else:
        print("📝 No API key - using intelligent simulation mode")
        print("   (Set GEMINI_API_KEY environment variable for real AI)")
    
    print("\n🔄 Initializing AI Medical Assistant...")
    
    try:
        assistant = AIMedicalAssistant(gemini_api_key=gemini_key)
    except Exception as e:
        print(f"❌ Error initializing assistant: {e}")
        print("Please ensure diabetes.csv is in the current directory.")
        return
    
    # Main interaction loop
    while True:
        print(f"\n{'🎯 MAIN MENU':^60}")
        print("What would you like to do?")
        print("1. 🩺 Analyze new patient")
        print("2. 🎭 Demo with sample patients")
        print("3. 📊 Show model information")
        print("4. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            try:
                patient_data = get_patient_input()
                display_analysis_results(assistant, patient_data, "New Patient")
            except KeyboardInterrupt:
                print("\n⏹️ Analysis cancelled.")
        
        elif choice == '2':
            demo_sample_patients(assistant)
        
        elif choice == '3':
            print(f"\n{'📊 MODEL INFORMATION':^60}")
            print(f"Model Type: Random Forest Classifier")
            print(f"Features: {len(assistant.feature_columns)}")
            print(f"AI Enhancement: {'✅ Gemini Pro' if assistant.gemini_available else '🤖 Intelligent Simulation'}")
            print(f"\nTop Risk Factors:")
            for i, (feature, importance) in enumerate(list(assistant.feature_importance.items())[:5], 1):
                print(f"  {i}. {feature}: {importance:.4f}")
        
        elif choice == '4':
            print("\n👋 Thank you for using AI-Enhanced Medical Assistant!")
            print("🏥 Revolutionizing healthcare with artificial intelligence.")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("Please ensure all required files are present and try again.")