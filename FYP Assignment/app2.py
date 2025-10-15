# ════════════════════════════════════════════════════════════════════════════
#  Rules Base Recommendation
# ════════════════════════════════════════════════════════════════════════════
from flask import Flask, render_template, request, redirect, url_for, flash, session, current_app
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, SelectField, StringField, PasswordField
from wtforms.validators import InputRequired, NumberRange
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import shap
import logging
from logging.handlers import RotatingFileHandler
import traceback



# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "FYP_RecoveryRatePrediction"
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# User model
class User(UserMixin):
    def __init__(self, id, username, password, role):
        self.id = id
        self.username = username
        self.password = password
        self.role = role  # Add the role attribute
        self.authenticated = False
    
    def is_active(self):
        return True
    
    def is_anonymous(self):
        return False
    
    def is_authenticated(self):
        return self.authenticated
    
    def get_id(self):
        return str(self.id)

users = {
    1: User(1, 'Hospital1', generate_password_hash('hospital1'), 'hospital'),
    2: User(2, 'Hospital2', generate_password_hash('hospital2'), 'hospital'),
    3: User(3, 'PolicyMaker1', generate_password_hash('policymaker1'), 'policymaker')
}



# Login manager to load user from user_id
@login_manager.user_loader
def load_user(user_id):
    return users.get(int(user_id))


FEATURE_ORDER = [
    "Income(USD)", "Year", "DALYs", "Mort_Rate%", "Prev_Rate%", "Doctors/1000", 
    "Urban_Rate%", "Health_Access%", "Avg_Cost(USD)", "Beds/1000", 
    "Disease_COVID-19", "Disease_HIV", "Disease_Influenza", "Disease_Zika", 
    "Country_Argentina", "Country_Australia", "Country_Brazil", "Country_Canada", 
    "Country_France", "Country_Germany", "Country_India", "Country_Indonesia", 
    "Country_Italy", "Country_Japan", "Country_Mexico", "Country_Nigeria", 
    "Country_Russia", "Country_Saudi Arabia", "Country_South Africa", 
    "Country_South Korea", "Country_Turkey", "Country_UK", "Country_USA"
]

# Load the trained model
model = joblib.load("xgb_model.pkl")
explainer = shap.TreeExplainer(model)

# Constants for scaling
Y_MIN, Y_MAX = 10.08, 99
PUB_MIN, PUB_MAX = 0, 100

# Function to scale outputs
def scaled_to_raw(x):
    return x * (Y_MAX - Y_MIN) + Y_MIN

def raw_to_public(x):
    normalized = (x - Y_MIN) / (Y_MAX - Y_MIN)
    return round(PUB_MIN + normalized * (PUB_MAX - PUB_MIN), 1)

# Function to preprocess input (feature ordering and handling missing features)
def preprocess_input(input_df):
    """Ensure all features are in correct order and handle missing columns"""
    processed = input_df.copy()

    # Ensure all features are in correct order
    for col in FEATURE_ORDER:
        if col not in processed.columns:
            processed[col] = 0  # Set missing features to 0
    
    return processed[FEATURE_ORDER]

# Define prediction form
class PredictionForm(FlaskForm):
    Income = FloatField("Income (USD) (e.g. 300-100000)", validators=[InputRequired()])
    Year = FloatField("Year (e.g. 2000-2025)", validators=[InputRequired()])
    DALYs = FloatField("DALYs (Disability-Adjusted Life Years) (e.g. 0-1800)", validators=[InputRequired()])
    Mort_Rate = FloatField("Mortality Rate (%) (e.g. 0-1)", validators=[InputRequired()])
    Prev_Rate = FloatField("Prevalence Rate (%) (e.g. 0-25)", validators=[InputRequired()])
    Doctors_per_1000 = FloatField("Doctors per 1000 people (e.g. 0-5)", validators=[InputRequired()])
    Urban_Rate = FloatField("Urbanization Rate (%) (e.g. 0-100)", validators=[InputRequired()])
    Health_Access = FloatField("Health Access (%) (e.g. 0-100)", validators=[InputRequired()])
    Avg_Cost = FloatField("Average Cost (USD) (e.g. 40-20000)", validators=[InputRequired()])
    Beds_per_1000 = FloatField("Beds per 1000 people (e.g. 0-20)", validators=[InputRequired()])
    
    disease_choices = [('COVID-19', 'COVID-19'), ('HIV', 'HIV'), 
                       ('Influenza', 'Influenza'), ('Zika', 'Zika')]
    disease = SelectField("Select Disease", choices=disease_choices, validators=[InputRequired()])
    
    country_choices = [
        ('Argentina', 'Argentina'), ('Australia', 'Australia'), ('Brazil', 'Brazil'),
        ('Canada', 'Canada'), ('France', 'France'), ('Germany', 'Germany'),
        ('India', 'India'), ('Indonesia', 'Indonesia'), ('Italy', 'Italy'),
        ('Japan', 'Japan'), ('Mexico', 'Mexico'), ('Nigeria', 'Nigeria'),
        ('Russia', 'Russia'), ('Saudi Arabia', 'Saudi Arabia'), 
        ('South Africa', 'South Africa'), ('South Korea', 'South Korea'),
        ('Turkey', 'Turkey'), ('UK', 'UK'), ('USA', 'USA')
    ]
    country = SelectField("Select Country", choices=country_choices, validators=[InputRequired()])
    
    submit = SubmitField("Predict")


# Helper functions for explanations

def generate_explanation_and_recommendation(score, shap_values, feature_names, form_data):
    # Create DataFrame with SHAP values and feature names
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values[0]  # Using the first (and only) prediction's SHAP values
    })
    
    # Sort by absolute SHAP value to get most important features
    shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
    top_features = shap_df.sort_values('abs_shap', ascending=False).head(5)
    
    explanation = ""
    recommendations = []
    
    # Base explanation based on score range
    if score >= 90:
        explanation = "Based on current health metrics and demographics, this patient has an excellent recovery rate."
        base_recommendation = "Encourage the patient to maintain their current healthy lifestyle and follow medical advice to sustain this positive trajectory."
    elif score >= 70:
        explanation = "Based on current health metrics and demographics, this patient has a good recovery rate."
        base_recommendation = "With some targeted improvements, the recovery rate could be further enhanced."
    elif score >= 50:
        explanation = "Based on current health metrics and demographics, this patient has an average recovery rate."
        base_recommendation = "Several areas could benefit from improvement to boost recovery prospects."
    else:
        explanation = "Based on current health metrics and demographics, this patient has a concerning recovery rate."
        base_recommendation = "Immediate attention to health metrics and demographics is required, along with assistance to improve them, in order to enhance recovery rate."
    
    recommendations.append(base_recommendation)
    
    # Detailed analysis based on top features
    feature_descriptions = {
        "Income(USD)": {
            "description": "Income level affects access to healthcare and the quality of care. Higher income helps to meet healthcare needs better.",
            "good": "Sufficient income to cover healthcare costs, ensuring timely access to necessary treatments.",
            "improve": "Consider exploring financial aid programs to reduce the burden of healthcare costs."
        },
        "Year": {
            "description": "Recent years may offer better treatment options due to advances in medicine and technology.",
            "good": "The current year benefits from the latest medical advancements, offering better chances for recovery.",
            "improve": "Look into the latest treatment options available for better outcomes."
        },
        "DALYs": {
            "description": "Disability-Adjusted Life Years measure the overall disease burden on health.",
            "good": "Low DALY means lower disease burden, indicating good health status.",
            "improve": "Focus on preventing complications and improving overall health to reduce disease burden."
        },
        "Mort_Rate%": {
            "description": "Mortality rate indicates how severe a disease is, with higher rates indicating more severe conditions.",
            "good": "Low mortality rate suggests the condition is less severe.",
            "improve": "Focus on prevention and improving healthcare to reduce the risk of death."
        },
        "Prev_Rate%": {
            "description": "Prevalence rate shows how common the condition is in the population.",
            "good": "A low prevalence rate means the condition is less common, making treatment more available.",
            "improve": "Look for treatment strategies used in areas with similar prevalence rates."
        },
        "Doctors/1000": {
            "description": "Availability of doctors affects access to healthcare services.",
            "good": "Good number of doctors available to provide quality healthcare.",
            "improve": "Consider seeking treatment in areas with better doctor-to-patient ratios if needed."
        },
        "Urban_Rate%": {
            "description": "Urbanization improves healthcare infrastructure, making it easier to access healthcare.",
            "good": "Living in urban areas offers better healthcare services and facilities.",
            "improve": "If you live in rural areas, consider temporarily relocating to urban areas for better healthcare."
        },
        "Health_Access%": {
            "description": "The percentage of people with access to healthcare.",
            "good": "Good healthcare access available, making it easier to get medical help.",
            "improve": "Work on improving access to healthcare, such as getting insurance or joining healthcare programs."
        },
        "Avg_Cost(USD)": {
            "description": "Treatment cost impacts affordability, and higher costs may affect decisions to seek care.",
            "good": "Treatment costs are manageable, allowing access to necessary care.",
            "improve": "Explore ways to save costs, such as using generic medications or applying for financial assistance programs."
        },
        "Beds/1000": {
            "description": "Hospital bed availability reflects the capacity of healthcare facilities to provide care.",
            "good": "Adequate number of hospital beds to cater to patients' needs.",
            "improve": "Consider seeking treatment in hospitals with better bed availability if needed."
        },
        "Disease_COVID-19": {
            "description": "COVID-19 factors, including how the virus spreads and available treatments, affect healthcare decisions.",
            "good": "",
            "improve": "Follow the latest COVID-19 guidelines and treatment protocols for better management of the disease."
        },
        "Disease_HIV": {
            "description": "HIV-specific factors, including adherence to treatment, directly affect the condition's progression.",
            "good": "",
            "improve": "Ensure you adhere to antiretroviral treatment to manage HIV and maintain good health."
        },
        "Disease_Influenza": {
            "description": "Influenza-specific factors, such as vaccination and antiviral treatments, help in controlling the disease.",
            "good": "",
            "improve": "Get vaccinated annually and use antiviral medications to reduce the severity of influenza."
        },
        "Disease_Zika": {
            "description": "Zika-specific factors, including mosquito control and prevention, are essential in controlling the disease.",
            "good": "",
            "improve": "Focus on mosquito control and preventive measures to reduce the risk of Zika infection."
        }
    }
    
    
    # Country specific advice
    developed_countries = ['Australia', 'Canada', 'France', 'Germany', 'Italy', 
                         'Japan', 'South Korea', 'UK', 'USA']
    current_country = form_data.get('country', '')
    if current_country in developed_countries:
        recommendations.append("Being in a developed country provides access to advanced treatments.")
    else:
        recommendations.append("Consider medical tourism to access better treatments if feasible.")
    
    # Analyze top features based on SHAP
    feature_analysis = []
    for _, row in top_features.iterrows():
        feature_name = row['feature']
        
        # Skip country features as we've already handled them
        if feature_name.startswith("Country_"):
            continue
            
        info = feature_descriptions.get(feature_name, {})
        direction = "increased" if row['shap_value'] > 0 else "decreased"
        magnitude = abs(row['shap_value'])
        
        analysis = f"• {feature_name}: {direction.capitalize()} recovery probability by {magnitude:.2f} points"
        
        if score < 90 and 'improve' in info:
            recommendations.append(f"For {feature_name}: {info['improve']}")
        
        feature_analysis.append(analysis)
    
    # Combine all explanations
    full_explanation = f"""
    <p>{explanation}</p>
    <p><strong>Top factors influencing this prediction:</strong></p>
    <ul>
        {"".join(f"<li>{fa}</li>" for fa in feature_analysis)}
    </ul>
    <p><small>SHAP values show how much each feature contributed to pushing the prediction higher or lower.</small></p>
    """
    
    # Add disease-specific advice
    current_disease = form_data.get('disease', '')
    if current_disease:
        disease_key = f"Disease_{current_disease}"
        if disease_key in feature_descriptions:
            recommendations.append(f"Disease-specific: {feature_descriptions[disease_key].get('improve', 'Follow standard treatment protocols for this condition')}")
    
    return {
        "explanation": full_explanation,
        "recommendations": recommendations
    }

from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import InputRequired

# Updated Metrics Form with disease-specific mortality rates
class MetricsForm(FlaskForm):
    # Disease-specific mortality rates
    Mort_Rate_COVID = FloatField("COVID-19 Mortality Rate (%)", validators=[InputRequired()])
    Mort_Rate_HIV = FloatField("HIV Mortality Rate (%)", validators=[InputRequired()])
    Mort_Rate_Flu = FloatField("Influenza Mortality Rate (%)", validators=[InputRequired()])
    Mort_Rate_Zika = FloatField("Zika Mortality Rate (%)", validators=[InputRequired()])
    
    # Other metrics
    Prev_Rate = FloatField("Prevalence Rate (%)", validators=[InputRequired()])
    Urban_Rate = FloatField("Urbanization Rate (%)", validators=[InputRequired()])
    Health_Access = FloatField("Health Access (%)", validators=[InputRequired()])
    submit = SubmitField("Update Metrics")

# Updated Metrics Data Storage with disease-specific rates
metrics_data = {
    "region1": {
        "Mort_Rate_COVID": 1.2,
        "Mort_Rate_HIV": 0.8,
        "Mort_Rate_Flu": 0.3,
        "Mort_Rate_Zika": 0.1,
        "Prev_Rate": 3.2,
        "Urban_Rate": 75,
        "Health_Access": 85,
        "last_updated": "2023-10-15"
    },
    "region2": {
        "Mort_Rate_COVID": 1.5,
        "Mort_Rate_HIV": 1.0,
        "Mort_Rate_Flu": 0.4,
        "Mort_Rate_Zika": 0.2,
        "Prev_Rate": 4.1,
        "Urban_Rate": 65,
        "Health_Access": 72,
        "last_updated": "2023-10-10"
    }
}

def generate_shap_plot(shap_values, features):
    plt.switch_backend('Agg')  # Ensure using non-GUI backend
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    plt.tight_layout()
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close(fig)  # Explicitly close the figure
    img_buf.seek(0)
    return base64.b64encode(img_buf.getvalue()).decode('utf-8')



# Login Form
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    submit = SubmitField('Login')

# Routes for login, logout, and dashboard
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    app.logger.debug(f"Form submitted: {form.is_submitted()}")
    app.logger.debug(f"Form validated: {form.validate_on_submit()}")
    
    if form.validate_on_submit():
        app.logger.debug("Form validated successfully")
        user = next((u for u in users.values() if u.username == form.username.data), None)
        app.logger.debug(f"User found: {user is not None}")
        
        if user and check_password_hash(user.password, form.password.data):
            app.logger.debug("Password matches")
            login_user(user)
            app.logger.debug(f"Current user authenticated: {current_user.is_authenticated}")
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            app.logger.debug("Invalid credentials")
            flash('Invalid username or password', 'danger')
    
    app.logger.debug(f"Form errors: {form.errors}")
    return render_template('login.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    recovery_score = current_user.recovery_score if current_user.recovery_score is not None else "Not Available"
    return render_template('dashboard.html', username=current_user.username, recovery_score=recovery_score)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    form = PredictionForm()
    score = None
    explanation = None
    recommendations = None
    shap_plot = None

    if form.validate_on_submit():
        try:
            # Prepare input data
            input_data = {
                "Income(USD)": form.Income.data,
                "Year": form.Year.data,
                "DALYs": form.DALYs.data,
                "Mort_Rate%": form.Mort_Rate.data,
                "Prev_Rate%": form.Prev_Rate.data,
                "Doctors/1000": form.Doctors_per_1000.data,
                "Urban_Rate%": form.Urban_Rate.data,
                "Health_Access%": form.Health_Access.data,
                "Avg_Cost(USD)": form.Avg_Cost.data,
                "Beds/1000": form.Beds_per_1000.data,
                f"Disease_{form.disease.data}": 1,
                f"Country_{form.country.data}": 1
            }

            # Initialize all disease and country features to 0
            for disease in ['COVID-19', 'HIV', 'Influenza', 'Zika']:
                if f"Disease_{disease}" not in input_data:
                    input_data[f"Disease_{disease}"] = 0
            
            for country in ['Argentina', 'Australia', 'Brazil', 'Canada', 'France', 
                          'Germany', 'India', 'Indonesia', 'Italy', 'Japan', 
                          'Mexico', 'Nigeria', 'Russia', 'Saudi Arabia', 
                          'South Africa', 'South Korea', 'Turkey', 'UK', 'USA']:
                if f"Country_{country}" not in input_data:
                    input_data[f"Country_{country}"] = 0

            # Convert to DataFrame and preprocess
            input_df = pd.DataFrame([input_data])
            processed_df = preprocess_input(input_df)

            # Make prediction
            scaled = model.predict(processed_df)[0]
            raw = scaled_to_raw(scaled)
            score = raw_to_public(raw)

            # Generate SHAP values and explanations
            shap_values = explainer.shap_values(processed_df)
            shap_plot = generate_shap_plot(shap_values, processed_df)

            form_data = {
                'Income': form.Income.data,
                'Year': form.Year.data,
                'DALYs': form.DALYs.data,
                'Mort_Rate': form.Mort_Rate.data,
                'Prev_Rate': form.Prev_Rate.data,
                'Doctors_per_1000': form.Doctors_per_1000.data,
                'Urban_Rate': form.Urban_Rate.data,
                'Health_Access': form.Health_Access.data,
                'Avg_Cost': form.Avg_Cost.data,
                'Beds_per_1000': form.Beds_per_1000.data,
                'disease': form.disease.data,
                'country': form.country.data
            }

            analysis = generate_explanation_and_recommendation(
                score=score,
                shap_values=shap_values,
                feature_names=processed_df.columns.tolist(),
                form_data=form_data
            )
            explanation = analysis['explanation']
            recommendations = analysis['recommendations']

            # Create SHAP plots
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, processed_df, plot_type="bar", show=False)
            plt.tight_layout()
            img_buf = BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight')
            img_buf.seek(0)
            shap_plot = base64.b64encode(img_buf.getvalue()).decode('utf-8')
            plt.close()


        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            flash("An error occurred during prediction. Please try again.", "error")
            return redirect(url_for('predict'))

    return render_template(
        "predict.html",
        form=form,
        score=score,
        explanation=explanation,
        recommendations=recommendations,
        shap_plot=shap_plot,
    )

# Home route
@app.route("/")
def home():
    return render_template("home.html")

from datetime import datetime
from flask import flash, redirect, render_template, url_for
from flask_login import login_required, current_user

@app.route("/metrics", methods=["GET", "POST"])
@login_required
def metrics():
    # Determine user's region (in a real app, this would come from user profile)
    user_region = "region1"  # Default region
    
    # Get current metrics for the user's region
    current_metrics = metrics_data.get(user_region, {})
    
    form = None
    if current_user.role == 'policymaker':
        form = MetricsForm()
        
        if form.validate_on_submit():
            # Update metrics for the region with disease-specific rates
            metrics_data[user_region] = {
                "Mort_Rate_COVID": form.Mort_Rate_COVID.data,
                "Mort_Rate_HIV": form.Mort_Rate_HIV.data,
                "Mort_Rate_Flu": form.Mort_Rate_Flu.data,
                "Mort_Rate_Zika": form.Mort_Rate_Zika.data,
                "Prev_Rate": form.Prev_Rate.data,
                "Urban_Rate": form.Urban_Rate.data,
                "Health_Access": form.Health_Access.data,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            flash('Metrics updated successfully!', 'success')
            return redirect(url_for('metrics'))
        
        # Pre-populate form with current values
        form.Mort_Rate_COVID.data = current_metrics.get('Mort_Rate_COVID', 0)
        form.Mort_Rate_HIV.data = current_metrics.get('Mort_Rate_HIV', 0)
        form.Mort_Rate_Flu.data = current_metrics.get('Mort_Rate_Flu', 0)
        form.Mort_Rate_Zika.data = current_metrics.get('Mort_Rate_Zika', 0)
        form.Prev_Rate.data = current_metrics.get('Prev_Rate', 0)
        form.Urban_Rate.data = current_metrics.get('Urban_Rate', 0)
        form.Health_Access.data = current_metrics.get('Health_Access', 0)
    
    return render_template("metrics.html", 
                         current_metrics=current_metrics,
                         form=form,
                         is_policymaker=current_user.role == 'policymaker',
                         user_region=user_region)

@app.route('/profile')
@login_required
def profile():
    # Fetch the user object
    user = current_user
    
    # Check the user role and render the appropriate profile content
    if user.role == 'hospital':
        return render_template('hospital_profile.html', username=user.username)
    elif user.role == 'policymaker':
        return render_template('policymaker_profile.html', username=user.username)
    else:
        # Default in case something goes wrong
        flash('Unknown user role', 'danger')
        return redirect(url_for('home'))

# Model Performance route
@app.route("/model-performance")
@login_required
def model_performance():
    return render_template("model_performance.html")

if __name__ == "__main__":
    app.run(debug=True)
