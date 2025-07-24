# 🚀 Setup Guide for Dr. AI - Diabetes Risk Assessment

## 1. Get Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

## 2. Configure Environment

**Option 1: Set Environment Variable (Recommended)**
```bash
# Windows
set GEMINI_API_KEY=your_actual_api_key_here

# macOS/Linux
export GEMINI_API_KEY=your_actual_api_key_here
```

**Option 2: Edit app.py directly**
```python
# In app.py, line 19, replace:
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-gemini-api-key-here')
# With:
GEMINI_API_KEY = 'your_actual_api_key_here'
```

## 3. Install Dependencies

```bash
pip install flask pandas numpy scikit-learn google-generativeai
```

## 4. Run the Application

```bash
python app.py
```

Then open: **http://127.0.0.1:5000**

## 🎯 How to Test the Conversational AI

Try these sample conversations:

### Basic Health Check
```
User: "Hi, I'm 35 years old and want to check my diabetes risk"
AI: Will ask follow-up questions about glucose, BMI, etc.
```

### Natural Information Sharing
```
User: "I'm a 45-year-old woman, I've been pregnant twice, my glucose level was 150 mg/dL last time I checked, and my BMI is around 28"
AI: Will gather remaining info and make prediction
```

### Educational Questions
```
User: "What factors increase diabetes risk?"
AI: Will provide educational information
```

### AI/ML Questions
```
User: "How does your AI prediction work?"
AI: Will explain the Random Forest model and PIMA dataset
```

## 🧠 How the AI Assistant Works

1. **Natural Language Processing**: Gemini API understands user messages
2. **Context Management**: Maintains conversation history
3. **Function Calling**: Automatically calls ML model when sufficient data is available
4. **Educational Mode**: Provides diabetes information and prevention tips

## 🔧 Features

- ✅ Conversational AI interface with Gemini
- ✅ Function calling for ML predictions
- ✅ Natural language data extraction
- ✅ Educational diabetes information
- ✅ Real-time chat with message history
- ✅ Professional medical UI design
- ✅ Automatic risk assessment
- ✅ Mobile-responsive design

## 📊 Sample Conversation Flow

```
👤 User: Hi, I want to check my diabetes risk

🩺 Dr. AI: Hello! I'd be happy to help assess your diabetes risk. I'll need some health information from you. Let's start with some basic details - could you tell me your age and if you've had any recent blood tests?

👤 User: I'm 42 years old, female, and my last glucose test was 145 mg/dL

🩺 Dr. AI: Thank you for that information. A glucose level of 145 mg/dL is slightly elevated (normal is under 140). To give you a comprehensive risk assessment, I'll need a few more details:

- What's your current BMI or height/weight?
- Do you know your blood pressure?
- Any family history of diabetes?
- Have you been pregnant before?

👤 User: I'm 5'6" and weigh 160 lbs, blood pressure around 130/80, my mom had diabetes, and I've had 2 pregnancies

🩺 Dr. AI: [Automatically calculates BMI and calls ML model]

Based on your information, here's your diabetes risk assessment:

**Risk Level: Moderate Risk (45% probability)**

Your assessment shows:
- Age: 42 years ✓
- BMI: 25.8 (slightly overweight)
- Glucose: 145 mg/dL (elevated)
- Blood pressure: 80 mmHg ✓
- Family history: Positive risk factor
- Pregnancies: 2

**Recommendations:**
- Consult with your healthcare provider about the elevated glucose
- Consider lifestyle modifications...
```

## ⚠️ Important Notes

- **API Key Security**: Never commit your API key to version control
- **Medical Disclaimer**: This is for educational purposes only
- **Rate Limits**: Gemini API has usage limits in free tier
- **Internet Required**: Application needs internet for Gemini API calls

## 🐛 Troubleshooting

**"Invalid API Key" Error:**
- Check your API key is correct
- Ensure it's properly set in environment variable

**"Model not found" Error:**
- Verify internet connection
- Check Gemini API service status

**Chat not responding:**
- Check browser console for errors
- Verify Flask server is running
- Check API key permissions