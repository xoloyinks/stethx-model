from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import requests
import json
import uvicorn
import re
from dotenv import dotenv_values

config = dotenv_values(".env")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# api key from environment variable or config file for security

OPENROUTER_API_KEY = config.get("OPENROUTER_API_KEY")

FEATURE_NAMES = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

GENDER_DECODE = {0: "Female", 1: "Male"}
SMOKING_HISTORY_DECODE = {
    4: "never",
    0: "no info",
    1: "current",
    3: "former",
    2: "ever",
    5: "not current"
}


filename = 'finalized_model.sav'
try:
    with open(filename, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    raise Exception(f"Error loading model: {e}")

class InputData(BaseModel):
    data: list[float]
    name: str

def format_input_data(input_list):
    if len(input_list) != len(FEATURE_NAMES):
        raise HTTPException(status_code=400, detail=f"Expected {len(FEATURE_NAMES)} features, got {len(input_list)}")
    
    if input_list[0] not in [0, 1]:
        raise HTTPException(status_code=400, detail=f"Invalid gender value: {input_list[0]}. Expected 0 (Female) or 1 (Male)")
    if input_list[4] not in [0, 1, 2, 3, 4, 5]:
        raise HTTPException(status_code=400, detail=f"Invalid smoking_history value: {input_list[4]}. Expected 0, 1, 2, 3, 4, or 5")
    
    patient_data = dict(zip(FEATURE_NAMES, input_list))
    
    input_array = np.asarray(input_list).reshape(1, -1)
    return input_array, patient_data

def decode_patient_data(patient_data):
    decoded_data = patient_data.copy()
    decoded_data['gender'] = GENDER_DECODE.get(patient_data['gender'], patient_data['gender'])
    decoded_data['smoking_history'] = SMOKING_HISTORY_DECODE.get(patient_data['smoking_history'], patient_data['smoking_history'])
    return decoded_data

# Function to construct the clinical decision support prompt
def build_prompt(patient_data, prediction):
    prediction_label = "Diabetic" if prediction == 1 else "Non-Diabetic"
    decoded_data = decode_patient_data(patient_data)
    print(prediction)
    prompt = f"""
You are an expert clinical decision support system specializing in diabetes prevention and management, grounded in evidence-based guidelines (e.g., American Diabetes Association, World Health Organization). You are provided with patient data (X variables) and a machine learning prediction (y variable) indicating whether the patient is diabetic or not:

*Patient Data (X variables):*
{json.dumps(decoded_data, indent=2)}

*ML Prediction (y variable):*
{prediction_label} ("Diabetic" or "Non-Diabetic")

*Task:*
Analyze the patient data and ML prediction to generate a clinical decision support response for medical practitioners in the specified JSON-like format. Identify the most significant X variable contributing to the diabetes diagnosis (for Diabetic) or future risk (for Non-Diabetic) based on clinical relevance and standard diabetes risk factors (e.g., fasting glucose, HbA1c, BMI). Provide clinical recommendations as an array of concise, actionable steps for management or prevention, tailored to the patient’s data. Include a rationale explaining why the key variable was selected, referencing clinical thresholds or guidelines. Include additional properties to enhance clinical utility, such as risk stratification, diagnostic tests, and interventions.

*Output Format:*

- If the patient is *Diabetic*:
{{
  "key_contributing_factor": "<X variable most strongly associated with the diabetes diagnosis>",
  "key_factor_rationale": "<Brief explanation of why this variable is the primary contributor, citing clinical thresholds or guidelines>",
  "clinical_recommendations": ["<Step 1: Specific action>", "<Step 2: Specific action>", "..."],
  "risk_level": "<High/Moderate/Low, based on severity and control>",
  "recommended_tests": ["<Specific follow-up tests, e.g., HbA1c, lipid panel>"],
  "therapeutic_interventions": ["<Targeted interventions, e.g., pharmacotherapy, lifestyle modifications>"],
  "diagnosis_code": "<ICD-10 code, e.g., E11.9>",
  "comorbid_risk_factors": ["<Other clinical factors impacting management, e.g., age, smoking status>"],
  "follow_up_interval": "<Time to reassessment or monitoring, e.g., 3 months>",
  "lifestyle_goals": ["<Concrete, guideline-aligned behavioral targets>"],
  "referrals": ["<Specialist referrals, e.g., endocrinologist, dietitian>"],
  "medication_considerations": ["<Age, renal, or polypharmacy considerations>"]
}}

- If the patient is *Non-Diabetic*:
{{
  "key_risk_factor": "<X variable posing the highest future diabetes risk>",
  "key_factor_rationale": "<Brief explanation of why this variable poses the highest risk, citing clinical thresholds or guidelines>",
  "clinical_recommendations": ["<Step 1: Specific action>", "<Step 2: Specific action>", "..."],
  "risk_level": "<High/Moderate/Low, based on risk factors>",
  "recommended_tests": ["<Screening tests, e.g., annual glucose test>"],
  "preventive_interventions": ["<Specific preventive actions, e.g., weight management, exercise>"],
  "comorbid_risk_factors": ["<Other risk-enhancing clinical factors>"],
  "follow_up_interval": "<Time to next screening or reassessment>",
  "lifestyle_goals": ["<Preventive behavior targets, e.g., exercise, diet>"],
  "referrals": ["<Referrals to preventive care or counseling>"]
}}

*Instructions:*
1. Prioritize the X variable with the strongest clinical impact (e.g., HbA1c ≥6.5% or fasting glucose ≥126 mg/dL for diagnosis; BMI ≥25 or family history for risk) per ADA/WHO guidelines.
2. In key_factor_rationale, explicitly reference clinical thresholds (e.g., HbA1c ≥6.5% per ADA) or guideline-based risk factors to justify the selection.
3. Provide clinical_recommendations as an array of 3-5 concise, actionable steps, each specifying a clinical action (e.g., initiate pharmacotherapy, refer to specialist, counsel on diet).
4. Assign risk_level using clinical thresholds (e.g., High for HbA1c ≥7% in Diabetics, or BMI ≥30 with family history in Non-Diabetics).
5. For Diabetic cases:
   - Assign the appropriate ICD-10 code (e.g., E11.9 for type 2 diabetes without complications).
   - List comorbid_risk_factors based on X variables or inferred conditions affecting management (e.g., age, hypertension).
   - Specify follow_up_interval for reassessment (e.g., 3 months for new diagnoses).
   - Define lifestyle_goals as concrete, guideline-aligned targets (e.g., reduce BMI by 5%).
   - Recommend referrals to relevant specialists (e.g., endocrinologist, dietitian).
   - Address medication_considerations, noting age, renal function, or polypharmacy risks.
6. For Non-Diabetic cases:
   - List comorbid_risk_factors based on X variables or inferred conditions increasing diabetes risk (e.g., smoking, obesity).
   - Specify follow_up_interval for next screening (e.g., 1 year for prediabetes).
   - Define lifestyle_goals as preventive, guideline-aligned targets (e.g., achieve BMI <25).
   - Recommend referrals to preventive care or counseling (e.g., dietitian, smoking cessation programs).
7. Recommend tests and interventions relevant to the patient’s profile, considering comorbidities and age.
8. If data is missing or ambiguous, use clinical judgment to make reasonable assumptions and note them in clinical_recommendations.
9. Use medical terminology suitable for practitioners, avoiding patient-facing language.
10. Ensure the response is a single, valid JSON object, formatted exactly as specified in the JSON-like structure, with no additional text or code block markers and ensure returned json match the appropriate prediction. Do not mix up the json (e.g., ```json).
"""
    return prompt

# Function to call OpenRouter API
def call_openrouter_api(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.5,
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        print(f"Raw API response: {content}") 
        
        # Remove code block markers and extra whitespace
        content = content.strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        
        # Extract the first valid JSON object using regex
        try:
            match = re.search(r'\{.*?\}(?=\s*$|\s*[\{\[])', content, re.DOTALL)
            if match:
                json_content = match.group(0)
                return json.loads(json_content)
            else:
                return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Content causing error: {content}")
            raise HTTPException(status_code=500, detail=f"Failed to parse API response: {e}")
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        raise HTTPException(status_code=500, detail=f"OpenRouter API request failed: {e}")


@app.post('/predict')
async def predict(input: InputData):
    try:
        print(f"Received input: {input.data}")  # Log input for debugging
        input_array, patient_data = format_input_data(input.data)
        prediction = model.predict(input_array)[0]
        confidence = model.predict_proba(input_array).max() if hasattr(model, "predict_proba") else None
        prompt = build_prompt(patient_data, prediction)
        clinical_response = call_openrouter_api(prompt)
        print(
            {
            "prediction": "Diabetic" if prediction == 1 else "Non-Diabetic",
            "confidence": confidence if confidence is not None else None,
            "name": input.name
        }
        )
        return {
            "prediction": "Diabetic" if prediction == 1 else "Non-Diabetic",
            "confidence": confidence if confidence is not None else None,
            "clinical_decision_support": clinical_response,
            "name": input.name
        }
    
    except HTTPException as e:
        print(f"HTTPException: {e.detail}")
        raise
    except Exception as e:
        print(f"General error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# if __name__ == "__main__":
    # uvicorn.run(app, host='localhost', port=5000)
    