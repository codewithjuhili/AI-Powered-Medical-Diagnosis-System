# Flask Medical AI Application

This is a Flask-based medical AI application converted from Streamlit. The application provides:

1. **User Authentication**: Login and Signup functionality
2. **Diagnosis Mode**: Predict risks for Diabetes, Blood Pressure, and Lung Cancer
3. **Training Mode**: Interactive quiz system with difficulty levels, progress tracking, and leaderboard

## Setup Instructions

### 1. Install Dependencies

```bash
pip install flask psycopg2-binary bcrypt pandas numpy scikit-learn joblib plotly
```

### 2. Database Setup

Make sure PostgreSQL is running and create a database:

```sql
CREATE DATABASE medical_ai;
```

Update database credentials in `app_flask.py` if needed:
```python
def get_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="medical_ai",
        user="postgres",
        password="12345678",
        port=5432
    )
```

### 3. Required Files

Ensure you have these files in your project directory:
- `disease_mcq_dataset_500.csv` - Quiz questions dataset
- `models/` folder with all trained models
- Static assets (images) in `static/` folder
- Templates in `templates/` folder

### 4. Run the Application

```bash
python app_flask.py
```

The application will be available at `http://localhost:5000`

## Features

### Diagnosis Mode
- Diabetes Risk Assessment
- Blood Pressure Risk Assessment
- Lung Cancer Risk Assessment
- Patient History Tracking

### Training Mode (Quiz)
- Three difficulty levels: Easy, Moderate, Hard
- Interactive quiz with 10 questions
- Real-time scoring and feedback
- Progress tracking
- Leaderboard

## Project Structure

```
Mini_Project_sem5/
├── app_flask.py              # Main Flask application
├── templates/                # HTML templates
│   ├── base.html
│   ├── login.html
│   ├── mode_selection.html
│   ├── diagnosis.html
│   ├── diagnosis_diabetes.html
│   ├── diagnosis_bp.html
│   ├── diagnosis_lung.html
│   ├── training_dashboard.html
│   ├── quiz_levels.html
│   ├── quiz_name_input.html
│   ├── quiz.html
│   ├── quiz_result.html
│   ├── training_progress.html
│   └── training_leaderboard.html
├── static/                   # Static files (CSS, images)
│   ├── style.css
│   ├── bg.png
│   └── ... (other images)
├── models/                   # Trained ML models
└── disease_mcq_dataset_500.csv  # Quiz dataset
```

## Usage

1. **Login/Signup**: Create an account or login with existing credentials
2. **Select Mode**: Choose between Diagnosis Mode or Training Mode
3. **Diagnosis Mode**: Select a disease, fill in patient details, get predictions
4. **Training Mode**: 
   - Select difficulty level
   - Enter your name
   - Take the quiz
   - View your progress and leaderboard

## Notes

- Session state is used to maintain quiz progress and user data
- Patient records are saved to PostgreSQL database
- Quiz progress is stored in Flask session (in-memory, will reset on server restart)
- All ML models are loaded at application startup

