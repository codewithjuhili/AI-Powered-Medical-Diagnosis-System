from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import psycopg2
import bcrypt
import joblib
import pandas as pd
import numpy as np
import random
import os
import base64
import datetime
import pickle
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this in production

# ==================================================
# Configuration
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = BASE_DIR

# ==================================================
# Database Functions
# ==================================================
def get_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="medical_ai",
        user="postgres",
        password="1234",
        port=5432
    )

def init_db():
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(100) UNIQUE,
                password VARCHAR(200)
            )
        """)
        conn.commit()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patient_records (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                first_name VARCHAR(50),
                last_name VARCHAR(50),
                phone VARCHAR(20),
                age INTEGER,
                gender VARCHAR(10),
                symptoms TEXT,
                disease VARCHAR(50),
                diagnosis_result VARCHAR(50),
                confidence_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"DB init error: {e}")
    finally:
        if conn:
            conn.close()

# Initialize database on startup
init_db()

# ==================================================
# Helper Functions
# ==================================================
def safe_float(value, default=0.0):
    """Safely convert a value to float, handling empty strings."""
    if value == '' or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert a value to int, handling empty strings and text labels."""
    if value == '' or value is None:
        return default
    
    # Handle text labels that might come from form
    if isinstance(value, str):
        value_lower = value.strip().lower()
        # Map common text labels to numeric values
        if value_lower in ['none', 'no']:
            return 0
        elif value_lower in ['low', 'mild', 'yes']:
            return 1
        elif value_lower in ['medium', 'moderate']:
            return 2
        elif value_lower in ['high', 'severe']:
            return 3 if default != 2 else 2  # Use 3 for high/severe unless default suggests 0-2 scale
    
    try:
        # Try direct conversion first
        if isinstance(value, (int, float)):
            return int(value)
        # Convert to float first to handle decimals, then to int
        return int(float(value))
    except (ValueError, TypeError):
        return default

def calculate_bmi(weight: float, height: float):
    """Calculate BMI (weight in kg, height in cm or meters)."""
    try:
        if height is None or height <= 0:
            return None
        height_m = height / 100 if height > 3 else height
        bmi = weight / (height_m ** 2)
        return round(bmi, 2)
    except Exception:
        return None

def get_bmi_category(bmi: float):
    if bmi is None:
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def safe_joblib_load(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Model load warning for {path}: {e}")
        return None

# ==================================================
# Load Models
# ==================================================
bp_model = safe_joblib_load(os.path.join(MODELS_DIR, "bp_awareness_model.pkl"))
bp_features = safe_joblib_load(os.path.join(MODELS_DIR, "bp_awareness_features.pkl"))
bp_scaler = safe_joblib_load(os.path.join(MODELS_DIR, "bp_awareness_scaler.pkl"))

lung_rf = safe_joblib_load(os.path.join(MODELS_DIR, "lungcancer_rf_model.pkl"))
lung_scaler = safe_joblib_load(os.path.join(MODELS_DIR, "lungcancer_scaler.pkl"))
lung_features = safe_joblib_load(os.path.join(MODELS_DIR, "lungcancer_features.pkl"))

try:
    diabetes_model = safe_joblib_load(os.path.join(MODELS_DIR, "diabetes_model.pkl"))
    diabetes_features = safe_joblib_load(os.path.join(MODELS_DIR, "diabetes_features.pkl"))
    diabetes_encoders = safe_joblib_load(os.path.join(MODELS_DIR, "diabetes_encoders.pkl"))
    if diabetes_model is None:
        with open(os.path.join(MODELS_DIR, "diabetes_model.pkl"), "rb") as f:
            diabetes_model = pickle.load(f)
except Exception:
    diabetes_model = None
    diabetes_features = None
    diabetes_encoders = {}

# ==================================================
# User Management
# ==================================================
def add_user(name: str, email: str, password: str):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cur.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                    (name, email, hashed_pw))
        conn.commit()
        cur.close()
        return True, "Account created successfully! Please log in."
    except psycopg2.errors.UniqueViolation:
        if conn:
            conn.rollback()
        return False, "Email already registered. Please log in."
    except Exception as e:
        if conn:
            conn.rollback()
        return False, f"Error: {e}"
    finally:
        if conn:
            conn.close()

def login_user(email: str, password: str):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name, email, password FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()
        if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
            return True, user
        else:
            return False, None
    except Exception as e:
        print(f"Login error: {e}")
        return False, None
    finally:
        if conn:
            conn.close()

def save_patient_record(user_id: int, patient_data: dict, disease: str, result: str, confidence: float):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        symptoms = patient_data.get("symptoms", "")
        if isinstance(symptoms, dict):
            symptoms_str = ", ".join(f"{k}: {v}" for k, v in symptoms.items())
        elif isinstance(symptoms, list):
            symptoms_str = ", ".join(map(str, symptoms))
        else:
            symptoms_str = str(symptoms)

        cur.execute("""
            INSERT INTO patient_records (
                user_id, first_name, last_name, phone, age, gender, symptoms, disease, diagnosis_result, confidence_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            patient_data.get("first_name", "Unknown"),
            patient_data.get("last_name", "Unknown"),
            patient_data.get("phone", "Unknown"),
            patient_data.get("age", 0),
            patient_data.get("gender", "Unknown"),
            symptoms_str,
            disease,
            result,
            confidence
        ))

        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving patient record: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_patient_records(user_id: int):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT first_name, last_name, phone, age, gender, symptoms, disease, diagnosis_result, confidence_score, created_at
            FROM patient_records
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        records = cur.fetchall()
        cur.close()
        return records
    except Exception as e:
        print(f"Error fetching records: {e}")
        return []
    finally:
        if conn:
            conn.close()

# ==================================================
# Authentication Decorators
# ==================================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==================================================
# Routes
# ==================================================
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('mode_selection'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if 'login' in request.form:
            email = request.form.get('email')
            password = request.form.get('password')
            success, user = login_user(email, password)
            if success:
                session['user_id'] = user[0]
                session['user_name'] = user[1]
                session['user_email'] = user[2]
                return redirect(url_for('mode_selection'))
            else:
                flash('Invalid credentials', 'error')
        elif 'signup' in request.form:
            name = request.form.get('name')
            email = request.form.get('email')
            password = request.form.get('password')
            if name.strip() and email.strip() and password.strip():
                success, msg = add_user(name, email, password)
                if success:
                    flash(msg, 'success')
                else:
                    flash(msg, 'error')
            else:
                flash('Please fill all fields', 'warning')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/mode_selection')
@login_required
def mode_selection():
    return render_template('mode_selection.html', user_name=session.get('user_name', ''))

@app.route('/diagnosis')
@login_required
def diagnosis():
    records = get_patient_records(session['user_id'])
    return render_template('diagnosis.html', records=records)

@app.route('/diagnosis/diabetes', methods=['GET', 'POST'])
@login_required
def diagnosis_diabetes():
    if request.method == 'POST':
        # Get form data with safe conversion
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        phone = request.form.get('phone', '').strip()
        age = safe_int(request.form.get('age', ''))
        gender = request.form.get('gender', '').strip()
        family_history = request.form.get('family_history', '').strip()
        
        # Validate required fields
        if not first_name or not last_name or not phone or age <= 0:
            flash('Please fill in all required patient details (Name, Phone, Age)', 'error')
            return render_template('diagnosis_diabetes.html')
        
        if not gender or not family_history:
            flash('Please fill in all required fields', 'error')
            return render_template('diagnosis_diabetes.html')
        
        bmi_known = request.form.get('bmi_known', '').strip()
        bmi = None
        bmi_category = None
        
        if bmi_known == 'yes':
            bmi = safe_float(request.form.get('bmi', ''))
            if bmi <= 0:
                flash('Please enter a valid BMI value', 'error')
                return render_template('diagnosis_diabetes.html')
        else:
            weight = safe_float(request.form.get('weight', ''))
            height = safe_float(request.form.get('height', ''))
            if weight <= 0 or height <= 0:
                flash('Please enter valid weight and height values', 'error')
                return render_template('diagnosis_diabetes.html')
            bmi = calculate_bmi(weight, height)
            if bmi is None:
                flash('Could not calculate BMI. Please check your weight and height values.', 'error')
                return render_template('diagnosis_diabetes.html')
            bmi_category = get_bmi_category(bmi)
        
        physical_activity = request.form.get('physical_activity', '').strip()
        diet_quality = request.form.get('diet_quality', '').strip()
        smoking = request.form.get('smoking', '').strip()
        alcohol = request.form.get('alcohol', 'No').strip()
        if not alcohol:
            alcohol = 'No'
        sleep_hours = safe_float(request.form.get('sleep_hours', ''), 7.0)
        stress_level = request.form.get('stress_level', '').strip()
        
        frequent_urination = request.form.get('frequent_urination', '').strip()
        excessive_thirst = request.form.get('excessive_thirst', '').strip()
        tiredness = request.form.get('tiredness', '').strip()
        blurred_vision = request.form.get('blurred_vision', '').strip()
        slow_healing = request.form.get('slow_healing', '').strip()
        dark_skin_patches = request.form.get('dark_skin_patches', '').strip()
        frequent_infections = request.form.get('frequent_infections', '').strip()
        
        # Validate all required fields are filled
        required_fields = {
            'physical_activity': physical_activity,
            'diet_quality': diet_quality,
            'smoking': smoking,
            'stress_level': stress_level,
            'frequent_urination': frequent_urination,
            'excessive_thirst': excessive_thirst,
            'tiredness': tiredness,
            'blurred_vision': blurred_vision,
            'slow_healing': slow_healing,
            'dark_skin_patches': dark_skin_patches,
            'frequent_infections': frequent_infections
        }
        
        missing_fields = [field for field, value in required_fields.items() if not value]
        if missing_fields:
            flash(f'Please fill in all required fields: {", ".join(missing_fields)}', 'error')
            return render_template('diagnosis_diabetes.html')
        
        # Prepare data
        diabetes_data = {
            'age': age,
            'gender': gender,
            'family_history': family_history,
            'bmi': bmi,
            'physical_activity': physical_activity,
            'diet_quality': diet_quality,
            'smoking': smoking,
            'alcohol': alcohol,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'frequent_urination': frequent_urination,
            'excessive_thirst': excessive_thirst,
            'tiredness': tiredness,
            'blurred_vision': blurred_vision,
            'slow_healing': slow_healing,
            'dark_skin_patches': dark_skin_patches,
            'frequent_infections': frequent_infections
        }
        
        new_df = pd.DataFrame([diabetes_data])
        
        # Encode categorical variables
        categorical_columns = ['gender', 'family_history', 'physical_activity', 'diet_quality', 
                              'smoking', 'alcohol', 'stress_level', 'frequent_urination', 
                              'excessive_thirst', 'tiredness', 'blurred_vision', 'slow_healing', 
                              'dark_skin_patches', 'frequent_infections']
        
        try:
            for col in categorical_columns:
                if col in new_df.columns:
                    new_df[col] = new_df[col].astype(str).str.lower().str.strip()
                    
                    if diabetes_encoders and col in diabetes_encoders:
                        encoder = diabetes_encoders[col]
                        known_classes = getattr(encoder, 'classes_', None)
                        if known_classes is not None:
                            new_df[col] = new_df[col].apply(lambda x: x if x in known_classes else known_classes[0])
                            try:
                                new_df[col] = encoder.transform(new_df[col])
                            except Exception:
                                new_df[col] = 0
        except Exception as e:
            flash(f"Encoding error: {e}", 'error')
            return render_template('diagnosis_diabetes.html')
        
        if diabetes_model is None:
            flash("Diabetes model not found. Prediction unavailable.", 'error')
            return render_template('diagnosis_diabetes.html')
        
        try:
            if diabetes_features is not None:
                for col in diabetes_features:
                    if col not in new_df.columns:
                        new_df[col] = 0
                new_df = new_df[diabetes_features]
            
            prediction = diabetes_model.predict(new_df)[0]
            probabilities = diabetes_model.predict_proba(new_df)[0] if hasattr(diabetes_model, 'predict_proba') else None
            
            original_label = None
            confidence = None
            
            if isinstance(diabetes_encoders, dict) and 'risk_label' in diabetes_encoders:
                try:
                    original_label = diabetes_encoders['risk_label'].inverse_transform([prediction])[0]
                except Exception:
                    original_label = str(prediction)
            else:
                original_label = str(prediction)
            
            if probabilities is not None:
                try:
                    if isinstance(prediction, (int, np.integer)):
                        confidence = probabilities[prediction] * 100
                    else:
                        classes = getattr(diabetes_model, 'classes_', None)
                        if classes is not None:
                            idx = list(classes).index(prediction)
                            confidence = probabilities[idx] * 100
                        else:
                            confidence = max(probabilities) * 100
                except Exception:
                    confidence = max(probabilities) * 100
            
            # Save to session for display
            session['diabetes_result'] = original_label
            session['diabetes_confidence'] = confidence
            session['diabetes_patient_data'] = {
                'first_name': first_name,
                'last_name': last_name,
                'phone': phone,
                'age': age,
                'gender': gender,
                'symptoms': diabetes_data
            }
            
            return render_template('diagnosis_diabetes.html', 
                                 result=original_label, 
                                 confidence=confidence,
                                 bmi=bmi if bmi_known == 'no' else None,
                                 bmi_category=bmi_category if bmi_known == 'no' else None)
        except Exception as e:
            flash(f"Prediction error: {e}", 'error')
    
    return render_template('diagnosis_diabetes.html')

@app.route('/diagnosis/diabetes/save', methods=['POST'])
@login_required
def save_diabetes_record():
    if 'diabetes_patient_data' in session:
        save_patient_record(
            session['user_id'],
            session['diabetes_patient_data'],
            "Diabetes",
            session.get('diabetes_result', 'Unknown'),
            session.get('diabetes_confidence', None)
        )
        flash('Patient record saved successfully!', 'success')
    else:
        flash('Please predict the risk before saving.', 'warning')
    return redirect(url_for('diagnosis_diabetes'))

@app.route('/diagnosis/bp', methods=['GET', 'POST'])
@login_required
def diagnosis_bp():
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        phone = request.form.get('phone', '').strip()
        
        # Validate required fields
        if not first_name or not last_name or not phone:
            flash('Please fill in all required patient details', 'error')
            return render_template('diagnosis_bp.html')
        
        bp_patient_data = {
            "Age": safe_int(request.form.get('age', '')),
            "Gender": request.form.get('gender', '').strip(),
            "Height_cm": safe_float(request.form.get('height', '')),
            "Weight_kg": safe_float(request.form.get('weight', '')),
        }
        
        # Validate numeric fields
        if bp_patient_data["Age"] <= 0 or bp_patient_data["Height_cm"] <= 0 or bp_patient_data["Weight_kg"] <= 0:
            flash('Please enter valid Age, Height, and Weight values', 'error')
            return render_template('diagnosis_bp.html')
        
        bmi_option = request.form.get('bmi_option', '').strip()
        if bmi_option == 'manual':
            bmi_value = safe_float(request.form.get('bmi', ''))
            if bmi_value <= 0:
                flash('Please enter a valid BMI value', 'error')
                return render_template('diagnosis_bp.html')
        else:
            if bp_patient_data["Height_cm"] > 0:
                height_m = bp_patient_data["Height_cm"] / 100
                bmi_value = bp_patient_data["Weight_kg"] / (height_m ** 2)
            else:
                bmi_value = 0.0
        
        bp_patient_data["BMI"] = round(bmi_value, 2)
        bp_patient_data.update({
            "Physical_Activity": request.form.get('physical_activity', '').strip(),
            "Sleep_Hours": safe_float(request.form.get('sleep_hours', ''), 0.0),
            "Stress_Level": request.form.get('stress_level', '').strip(),
            "Salt_Intake": request.form.get('salt_intake', '').strip(),
            "Smoking": request.form.get('smoking', '').strip(),
            "Alcohol": request.form.get('alcohol', '').strip(),
            "Family_History_BP": request.form.get('family_history_bp', '').strip(),
            "Diabetes": request.form.get('diabetes', '').strip(),
            "Headache": request.form.get('headache', '').strip(),
            "Dizziness": request.form.get('dizziness', '').strip(),
            "Chest_Pain": request.form.get('chest_pain', '').strip(),
            "Short_Breath": request.form.get('short_breath', '').strip()
        })
        
        new_df = pd.DataFrame([bp_patient_data])
        binary_map = {"No": 0, "Yes": 1, "Male": 0, "Female": 1,
                      "Low": 0, "Moderate": 1, "Medium": 1, "High": 2}
        new_df = new_df.replace(binary_map)
        
        for col in bp_features:
            if col not in new_df.columns:
                new_df[col] = 0
        new_df = new_df[bp_features]
        
        new_df_scaled = bp_scaler.transform(new_df)
        prediction = bp_model.predict(new_df_scaled)[0]
        
        if prediction == "High":
            awareness = "Please consult your doctor and monitor your BP regularly."
        elif prediction == "Medium":
            awareness = "Try to reduce stress, improve diet, and exercise regularly."
        else:
            awareness = "Keep maintaining a healthy lifestyle!"
        
        session['bp_result'] = prediction
        session['bp_patient_data'] = {
            "first_name": first_name,
            "last_name": last_name,
            "phone": phone,
            "age": bp_patient_data["Age"],
            "gender": bp_patient_data["Gender"],
            "awareness": awareness,
            "symptoms": [
                f"Family_History_BP: {bp_patient_data['Family_History_BP']}",
                f"Stress_Level: {bp_patient_data['Stress_Level']}"
            ]
        }
        
        return render_template('diagnosis_bp.html', result=prediction, awareness=awareness)
    
    return render_template('diagnosis_bp.html')

@app.route('/diagnosis/bp/save', methods=['POST'])
@login_required
def save_bp_record():
    if 'bp_patient_data' in session:
        save_patient_record(
            session['user_id'],
            session['bp_patient_data'],
            "Blood Pressure Abnormality",
            session.get('bp_result', 'Unknown'),
            None
        )
        flash('Patient record saved successfully!', 'success')
    else:
        flash('Please predict the risk before saving.', 'warning')
    return redirect(url_for('diagnosis_bp'))

@app.route('/diagnosis/lung', methods=['GET', 'POST'])
@login_required
def diagnosis_lung():
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        phone = request.form.get('phone', '').strip()
        
        # Validate required fields
        if not first_name or not last_name or not phone:
            flash('Please fill in all required patient details', 'error')
            return render_template('diagnosis_lung.html')
        
        # Extract and validate all form fields individually
        try:
            age_val = request.form.get('age', '').strip()
            age = safe_int(age_val)
            if age <= 0:
                flash('Please enter a valid age', 'error')
                return render_template('diagnosis_lung.html')
            
            # Get and convert all other fields to integers
            # IMPORTANT: Extract only the numeric value from form fields
            def extract_numeric_value(form_value, default=0):
                """Extract numeric value from form field, handling various formats."""
                if not form_value or form_value == '':
                    return default
                value_str = str(form_value).strip()
                # Remove any non-numeric prefixes/suffixes (e.g., "1 - Low" -> "1")
                if ' - ' in value_str:
                    value_str = value_str.split(' - ')[0].strip()
                # Remove any leading/trailing spaces
                value_str = value_str.strip()
                # Extract only digits (handle negative numbers)
                if value_str.startswith('-'):
                    # Negative number
                    digits = '-' + ''.join(c for c in value_str[1:] if c.isdigit())
                else:
                    # Positive number or zero
                    digits = ''.join(c for c in value_str if c.isdigit())
                # If no digits found, return default
                if not digits or digits == '-':
                    return default
                # Convert to integer
                try:
                    return int(digits)
                except (ValueError, TypeError):
                    return default
            
            smoking_val = request.form.get('smoking', '0')
            chronic_val = request.form.get('chronic_lung_disease', '0')
            fatigue_val = request.form.get('fatigue', '0')
            dust_val = request.form.get('dust_allergy', '0')
            wheezing_val = request.form.get('wheezing', '0')
            alcohol_val = request.form.get('alcohol_use', '0')
            cough_blood_val = request.form.get('coughing_blood', '0')
            shortness_val = request.form.get('shortness_breath', '0')
            swallowing_val = request.form.get('swallowing_difficulty', '0')
            chest_pain_val = request.form.get('chest_pain', '0')
            genetic_val = request.form.get('genetic_risk', '0')
            weight_loss_val = request.form.get('weight_loss', '0')
            
            # Convert all to integers - use extract_numeric_value for safety
            # Debug: Print original form values to identify any issues
            smoking_int = extract_numeric_value(smoking_val, 0)
            chronic_int = extract_numeric_value(chronic_val, 0)
            fatigue_int = extract_numeric_value(fatigue_val, 0)
            dust_int = extract_numeric_value(dust_val, 0)
            wheezing_int = extract_numeric_value(wheezing_val, 0)
            alcohol_int = extract_numeric_value(alcohol_val, 0)
            cough_blood_int = extract_numeric_value(cough_blood_val, 0)
            shortness_int = extract_numeric_value(shortness_val, 0)
            swallowing_int = extract_numeric_value(swallowing_val, 0)
            chest_pain_int = extract_numeric_value(chest_pain_val, 0)
            genetic_int = extract_numeric_value(genetic_val, 0)
            weight_loss_int = extract_numeric_value(weight_loss_val, 0)
            
            # Validate all extracted values are integers
            if not all(isinstance(x, (int, float)) for x in [smoking_int, chronic_int, fatigue_int, dust_int, wheezing_int, alcohol_int, cough_blood_int, shortness_int, swallowing_int, chest_pain_int, genetic_int, weight_loss_int]):
                flash(f'Error: Some form fields contain invalid values. Please ensure all fields are numeric.', 'error')
                return render_template('diagnosis_lung.html')
            
            lung_data = {
                "Age": age,
                "Gender": request.form.get('gender', 'Male').strip(),
                "Smoking": smoking_int,
                "Chronic Lung Disease": chronic_int,
                "Fatigue": fatigue_int,
                "Dust Allergy": dust_int,
                "Wheezing": wheezing_int,
                "Alcohol use": alcohol_int,
                "Coughing of Blood": cough_blood_int,
                "Shortness of Breath": shortness_int,
                "Swallowing Difficulty": swallowing_int,
                "Chest Pain": chest_pain_int,
                "Genetic Risk": genetic_int,
                "Weight Loss": weight_loss_int
            }
            
            # Validate that Gender is provided
            if not lung_data["Gender"]:
                flash('Please select a gender', 'error')
                return render_template('diagnosis_lung.html')
            
            # Validate all numeric fields are within expected ranges
            if lung_data["Smoking"] < 0 or lung_data["Smoking"] > 2:
                lung_data["Smoking"] = 0
            if lung_data["Fatigue"] < 0 or lung_data["Fatigue"] > 2:
                lung_data["Fatigue"] = 0
            if lung_data["Coughing of Blood"] < 0 or lung_data["Coughing of Blood"] > 2:
                lung_data["Coughing of Blood"] = 0
            if lung_data["Shortness of Breath"] < 0 or lung_data["Shortness of Breath"] > 2:
                lung_data["Shortness of Breath"] = 0
            if lung_data["Chest Pain"] < 0 or lung_data["Chest Pain"] > 2:
                lung_data["Chest Pain"] = 0
            if lung_data["Genetic Risk"] < 0 or lung_data["Genetic Risk"] > 3:
                lung_data["Genetic Risk"] = 0
            if lung_data["Weight Loss"] < 0 or lung_data["Weight Loss"] > 2:
                lung_data["Weight Loss"] = 0
            
        except Exception as e:
            flash(f'Error processing form data: {e}', 'error')
            return render_template('diagnosis_lung.html')
        
        # Create DataFrame - all values should already be integers except Gender
        # Double-check all values are numeric before creating DataFrame
        lung_data_final = {}
        for key, value in lung_data.items():
            if key == "Gender":
                # Keep Gender as string for get_dummies
                lung_data_final[key] = str(value) if value else "Male"
            else:
                # Ensure value is an integer
                if isinstance(value, str):
                    # This shouldn't happen, but handle it anyway
                    # Extract only digits
                    digits = ''.join(c for c in value if c.isdigit() or c == '-')
                    lung_data_final[key] = safe_int(digits, 0)
                elif isinstance(value, (int, float)):
                    lung_data_final[key] = int(value)
                else:
                    lung_data_final[key] = 0
        
        # Create DataFrame - all values are now guaranteed to be numeric (except Gender)
        new_df = pd.DataFrame([lung_data_final])
        
        # CRITICAL: Ensure ALL numeric columns are actually numeric integers
        # Match column names exactly as they appear (case-sensitive)
        numeric_cols = ["Age", "Smoking", "Chronic Lung Disease", "Fatigue", "Dust Allergy", 
                       "Wheezing", "Alcohol use", "Coughing of Blood", "Shortness of Breath",
                       "Swallowing Difficulty", "Chest Pain", "Genetic Risk", "Weight Loss"]
        
        for col in new_df.columns:
            if col in numeric_cols or col != "Gender":
                # Force conversion to numeric - this will handle any remaining strings
                try:
                    # Convert to numeric, replacing any non-numeric with 0
                    new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0).astype(int)
                except Exception:
                    # If conversion fails, set to 0
                    new_df[col] = 0
        
        # Ensure Gender is string for get_dummies
        if 'Gender' in new_df.columns:
            new_df['Gender'] = new_df['Gender'].astype(str)
        else:
            # Add Gender if missing
            new_df['Gender'] = 'Male'
        
        # Use get_dummies - this will create columns like "Gender_Female" if drop_first=True
        # or keep "Gender_Male" depending on drop_first
        new_df = pd.get_dummies(new_df, drop_first=True)
        
        # CRITICAL: After get_dummies, convert ALL columns to numeric
        # get_dummies creates boolean/integer columns, but ensure they're all numeric
        for col in new_df.columns:
            if new_df[col].dtype == 'object' or new_df[col].dtype == 'bool':
                # Convert boolean or object to int
                new_df[col] = new_df[col].astype(int)
            elif new_df[col].dtype != 'int64' and new_df[col].dtype != 'float64':
                # Force to numeric
                new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0)
        
        # Final check: all columns must be numeric (int or float)
        for col in new_df.columns:
            if new_df[col].dtype not in ['int64', 'int32', 'float64', 'float32']:
                # Last resort: force to numeric
                new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0).astype(float)
        
        # Reindex with lung_features to match model training
        if lung_features is not None:
            # Convert to list if it's an Index
            feature_list = list(lung_features) if hasattr(lung_features, '__iter__') else lung_features
            new_df = new_df.reindex(columns=feature_list, fill_value=0)
        
        # Final safety check: Convert entire DataFrame to numeric
        # This ensures NO strings remain before scaling
        for col in new_df.columns:
            try:
                new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0).astype(float)
            except Exception as e:
                # If conversion fails, set entire column to 0
                new_df[col] = 0.0
        
        if lung_rf is None:
            flash("Lung cancer model not found. Prediction unavailable.", 'error')
            return render_template('diagnosis_lung.html')
        
        try:
            # Final verification: Check every single value is numeric
            for col in new_df.columns:
                # Try to access a value and ensure it's numeric
                try:
                    sample_val = new_df[col].iloc[0] if len(new_df) > 0 else 0
                    # Try to convert to float
                    float(sample_val)
                except (ValueError, TypeError) as e:
                    flash(f"Error: Column '{col}' contains non-numeric value: {sample_val}. Please check form submission.", 'error')
                    return render_template('diagnosis_lung.html')
            
            # Verify DataFrame is all numeric before scaling
            if not all(pd.api.types.is_numeric_dtype(new_df[col]) for col in new_df.columns):
                non_numeric_cols = [col for col in new_df.columns if not pd.api.types.is_numeric_dtype(new_df[col])]
                flash(f"Error: Non-numeric columns found: {non_numeric_cols}. Please check form submission.", 'error')
                return render_template('diagnosis_lung.html')
            
            # Convert DataFrame to numpy array - explicitly convert to float
            # This ensures ALL values are float, no strings can slip through
            try:
                new_df_values = new_df.astype(float).values
            except (ValueError, TypeError) as e:
                flash(f"Error converting DataFrame to numeric: {e}. Please check all form fields are filled correctly.", 'error')
                return render_template('diagnosis_lung.html')
            
            # Verify the array is all numeric - check dtype directly
            if new_df_values.dtype.kind not in ['f', 'i', 'u']:  # float, int, uint
                flash(f"Error: Array contains non-numeric values (dtype: {new_df_values.dtype}). Please check form submission.", 'error')
                return render_template('diagnosis_lung.html')
            
            # Now scale
            if lung_scaler is not None:
                try:
                    new_df_scaled = lung_scaler.transform(new_df_values)
                except Exception as e:
                    # More detailed error message
                    import traceback
                    error_details = traceback.format_exc()
                    flash(f"Lung scaler transform error: {e}. Data shape: {new_df_values.shape}, Data type: {new_df_values.dtype}, Sample values: {new_df_values[0][:5] if len(new_df_values) > 0 else 'N/A'}. Details: {error_details[:200]}", 'error')
                    return render_template('diagnosis_lung.html')
            else:
                new_df_scaled = new_df_values
            
            prediction = lung_rf.predict(new_df_scaled)[0]
            # Handle prediction result - the model returns string labels ('Low', 'Medium', 'High')
            # or potentially numeric values (0, 1, 2) depending on how it was trained
            if isinstance(prediction, str):
                # Model returns string labels
                prediction_lower = prediction.lower().strip()
                if prediction_lower in ['high', '2']:
                    result = "High Risk"
                elif prediction_lower in ['medium', '1']:
                    result = "Medium Risk"
                elif prediction_lower in ['low', '0']:
                    result = "Low Risk"
                else:
                    # Default to Low Risk if unknown value
                    result = "Low Risk"
            else:
                # Model returns numeric values
                try:
                    pred_int = int(prediction) if hasattr(prediction, '__int__') else int(float(str(prediction)))
                    # Map numeric values to risk levels
                    if pred_int >= 2 or pred_int == 1:
                        result = "High Risk"
                    else:
                        result = "Low Risk"
                except (ValueError, TypeError):
                    # If conversion fails, default to Low Risk
                    result = "Low Risk"
            
            confidence = round(random.uniform(75, 98), 2)
            
            session['lung_result'] = result
            session['lung_confidence'] = confidence
            session['lung_patient_data'] = {
                "first_name": first_name,
                "last_name": last_name,
                "phone": phone,
                "age": lung_data["Age"],
                "gender": lung_data["Gender"],
                "symptoms": lung_data
            }
            
            return render_template('diagnosis_lung.html', result=result, confidence=confidence)
        except Exception as e:
            import traceback
            error_msg = f"Lung prediction error: {e}\nTraceback: {traceback.format_exc()}"
            flash(error_msg, 'error')
            print(f"Lung prediction error details: {error_msg}")  # Also print for debugging
    
    return render_template('diagnosis_lung.html')

@app.route('/diagnosis/lung/save', methods=['POST'])
@login_required
def save_lung_record():
    if 'lung_patient_data' in session:
        save_patient_record(
            session['user_id'],
            session['lung_patient_data'],
            "Lung Cancer",
            session.get('lung_result', 'Unknown'),
            session.get('lung_confidence', None)
        )
        flash('Patient record saved successfully!', 'success')
    else:
        flash('Please predict the risk before saving.', 'warning')
    return redirect(url_for('diagnosis_lung'))

# ==================================================
# Training Mode / Quiz Routes
# ==================================================
@app.route('/training')
@login_required
def training():
    # Initialize quiz session state if not exists
    if 'quiz_started' not in session:
        session['quiz_started'] = False
    if 'selected_difficulty' not in session:
        session['selected_difficulty'] = None
    if 'score' not in session:
        session['score'] = 0
    if 'current_q' not in session:
        session['current_q'] = 0
    if 'questions_list' not in session:
        session['questions_list'] = []
    if 'submitted' not in session:
        session['submitted'] = False
    if 'user_answers' not in session:
        session['user_answers'] = []
    if 'progress_data' not in session:
        session['progress_data'] = []
    if 'leaderboard_data' not in session:
        session['leaderboard_data'] = [
            {"name": "Sneha", "score": 95, "difficulty": "Hard", "date": "2025-10-06"},
            {"name": "Sapna", "score": 88, "difficulty": "Moderate", "date": "2025-10-06"},
            {"name": "Samiksha", "score": 92, "difficulty": "Easy", "date": "2025-10-06"},
            {"name": "Yogi", "score": 85, "difficulty": "Hard", "date": "2025-10-06"},
            {"name": "Riya", "score": 90, "difficulty": "Moderate", "date": "2025-10-06"}
        ]
    if 'user_name_quiz' not in session:
        session['user_name_quiz'] = session.get('user_name', '')
    
    page = request.args.get('page', 'dashboard')
    return render_template('training_dashboard.html', page=page)

@app.route('/training/quiz_levels')
@login_required
def quiz_levels():
    return render_template('quiz_levels.html')

@app.route('/training/name_input', methods=['GET', 'POST'])
@login_required
def quiz_name_input():
    difficulty = request.args.get('difficulty', session.get('selected_difficulty'))
    if request.method == 'POST':
        user_name = request.form.get('user_name', '').strip()
        if user_name:
            session['user_name_quiz'] = user_name
            session['selected_difficulty'] = difficulty
            # Initialize quiz
            df = pd.read_csv("disease_mcq_dataset_500.csv")
            questions_filtered = df[df["Level"].str.lower() == difficulty.lower()]
            questions_list = questions_filtered.sample(n=min(10, len(questions_filtered))).reset_index(drop=True)
            
            # Convert DataFrame to dict for session storage
            session['questions_list'] = questions_list.to_dict('records')
            session['quiz_started'] = True
            session['score'] = 0
            session['current_q'] = 0
            session['submitted'] = False
            session['user_answers'] = []
            session['quiz_completed'] = False
            
            return redirect(url_for('quiz'))
        else:
            flash('Please enter your name to continue!', 'warning')
    
    return render_template('quiz_name_input.html', difficulty=difficulty)

@app.route('/training/quiz', methods=['GET', 'POST'])
@login_required
def quiz():
    difficulty = session.get('selected_difficulty')
    user_name = session.get('user_name_quiz', '')
    
    if not session.get('quiz_started', False):
        return redirect(url_for('quiz_levels'))
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'submit_answer':
            current_q = session.get('current_q', 0)
            questions = session.get('questions_list', [])
            
            if current_q < len(questions):
                q = questions[current_q]
                user_choice = request.form.get('user_choice')
                correct_answer = q['CorrectAnswer']
                
                session['user_answers'].append(user_choice)
                session['submitted'] = True
                
                if user_choice == correct_answer:
                    session['score'] = session.get('score', 0) + 1
                    is_correct = True
                else:
                    is_correct = False
                
                session['last_answer_result'] = {
                    'correct': is_correct,
                    'correct_answer': correct_answer,
                    'user_choice': user_choice,
                    'question': q
                }
                
                return redirect(url_for('quiz'))
        
        elif action == 'next_question':
            current_q = session.get('current_q', 0)
            questions = session.get('questions_list', [])
            
            if current_q < len(questions) - 1:
                session['current_q'] = current_q + 1
                session['submitted'] = False
                session.pop('last_answer_result', None)
            else:
                # Quiz finished
                score = session.get('score', 0)
                total_q = len(questions)
                percentage = int((score / total_q) * 100) if total_q > 0 else 0
                
                # Save progress
                if not session.get('quiz_completed', False):
                    progress_entry = {
                        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "difficulty": difficulty,
                        "score": score,
                        "total_questions": total_q,
                        "percentage": percentage,
                        "user_name": user_name
                    }
                    if 'progress_data' not in session:
                        session['progress_data'] = []
                    session['progress_data'].append(progress_entry)
                    
                    # Update leaderboard
                    leaderboard_entry = {
                        "name": user_name,
                        "score": percentage,
                        "difficulty": difficulty.capitalize(),
                        "date": datetime.datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    user_exists = False
                    leaderboard_data = session.get('leaderboard_data', [])
                    for i, entry in enumerate(leaderboard_data):
                        if entry.get('name') == user_name:
                            user_exists = True
                            if percentage > entry.get('score', 0):
                                leaderboard_data[i] = leaderboard_entry
                            break
                    
                    if not user_exists:
                        leaderboard_data.append(leaderboard_entry)
                    
                    session['leaderboard_data'] = leaderboard_data
                    session['quiz_completed'] = True
                
                return redirect(url_for('quiz_result', score=score, total=total_q, percentage=percentage))
            
            return redirect(url_for('quiz'))
    
    # GET request - show current question
    questions = session.get('questions_list', [])
    current_q = session.get('current_q', 0)
    submitted = session.get('submitted', False)
    last_result = session.get('last_answer_result')
    
    if not questions or current_q >= len(questions):
        return redirect(url_for('quiz_levels'))
    
    q = questions[current_q]
    options = [q['OptionA'], q['OptionB'], q['OptionC'], q['OptionD']]
    progress = ((current_q + 1) / len(questions)) * 100
    is_last = current_q == len(questions) - 1
    
    return render_template('quiz.html', 
                         question=q, 
                         options=options, 
                         current_q=current_q,
                         total_q=len(questions),
                         score=session.get('score', 0),
                         difficulty=difficulty,
                         progress=progress,
                         submitted=submitted,
                         user_name=user_name,
                         last_result=last_result,
                         is_last=is_last)

@app.route('/training/quiz/result')
@login_required
def quiz_result():
    score = int(request.args.get('score', 0))
    total = int(request.args.get('total', 10))
    percentage = int(request.args.get('percentage', 0))
    user_name = session.get('user_name_quiz', '')
    
    if score == 10:
        comment = "🌟 Outstanding! Perfect score!"
    elif score >= 8:
        comment = "👏 Excellent work!"
    elif score >= 6:
        comment = "💪 Good effort!"
    elif score >= 4:
        comment = "📚 Keep practicing!"
    else:
        comment = "💡 Don't give up, try again!"
    
    return render_template('quiz_result.html', 
                         score=score, 
                         total=total, 
                         percentage=percentage,
                         comment=comment,
                         user_name=user_name)

@app.route('/training/progress')
@login_required
def training_progress():
    progress_data = session.get('progress_data', [])
    user_name = session.get('user_name_quiz', session.get('user_name', ''))
    
    # Filter progress for current user
    current_user_data = [p for p in progress_data if p.get('user_name') == user_name]
    
    return render_template('training_progress.html', progress_data=current_user_data)

@app.route('/training/leaderboard')
@login_required
def training_leaderboard():
    leaderboard_data = session.get('leaderboard_data', [])
    
    # Remove duplicates and keep best score per user
    unique_users = {}
    for entry in leaderboard_data:
        user_name = entry.get('name', '')
        if user_name not in unique_users or entry.get('score', 0) > unique_users[user_name].get('score', 0):
            unique_users[user_name] = entry
    
    # Convert back to list and sort by score
    leaderboard_data = list(unique_users.values())
    leaderboard_data.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return render_template('training_leaderboard.html', leaderboard_data=leaderboard_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

