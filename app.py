from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load the trained model
MODEL_PATH = 'random_forest_model_forplant.joblib'
DATA_PATH = 'plant_health_data.csv'
model = None
data_sample = None

# Define the expected feature columns based on your CSV
FEATURE_COLUMNS = [
    'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature', 'Humidity',
    'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', 'Phosphorus_Level',
    'Potassium_Level', 'Chlorophyll_Content', 'Electrochemical_Signal'
]

def load_model():
    """Load the Random Forest model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            print(f"❌ Model file not found: {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def load_sample_data():
    """Load sample data to understand data ranges"""
    global data_sample
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            data_sample = df[FEATURE_COLUMNS].describe()
            print(f"✅ Sample data loaded from {DATA_PATH}")
            print(f"📊 Data shape: {df.shape}")
            return True
        else:
            print(f"⚠️  Sample data file not found: {DATA_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error loading sample data: {str(e)}")
        return False

def validate_sensor_data(data):
    """Validate and preprocess sensor data"""
    try:
        # Convert to pandas DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Check for required columns
        missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
        if missing_cols:
            return None, f"Missing required columns: {missing_cols}"
        
        # Select only required features in correct order
        feature_data = df[FEATURE_COLUMNS]
        
        # Check for missing values
        if feature_data.isnull().any().any():
            return None, "Missing values detected in sensor data"
        
        # Convert to numpy array
        features = feature_data.values
        
        return features, None
        
    except Exception as e:
        return None, f"Data validation error: {str(e)}"

def interpret_prediction(prediction, probabilities=None):
    """Convert model prediction to human-readable status"""
    try:
        # Handle different prediction formats
        if isinstance(prediction, (list, np.ndarray)):
            pred_value = prediction[0]
        else:
            pred_value = prediction
        
        # Map prediction to status (adjust based on your model's classes)
        status_mapping = {
            'High Stress': {
                'status': 'high_stress',
                'message': '🚨 High Stress Detected',
                'description': 'The plant is experiencing significant stress. Immediate attention required.',
                'color_class': 'diseased',
                'risk_level': 'High',
                'priority': 'Critical'
            },
            'Moderate Stress': {
                'status': 'moderate_stress',
                'message': '⚠️ Moderate Stress Detected',
                'description': 'The plant shows signs of stress. Monitor closely and adjust care routine.',
                'color_class': 'warning',
                'risk_level': 'Medium',
                'priority': 'Medium'
            },
            'Low Stress': {
                'status': 'low_stress',
                'message': '🟡 Low Stress Detected',
                'description': 'Minor stress indicators present. Continue monitoring.',
                'color_class': 'warning',
                'risk_level': 'Low',
                'priority': 'Low'
            },
            'Healthy': {
                'status': 'healthy',
                'message': '🌱 Plant is Healthy',
                'description': 'All sensor readings are within optimal ranges. Plant appears healthy.',
                'color_class': 'healthy',
                'risk_level': 'Low',
                'priority': 'Normal'
            }
        }
        
        # Try to match the prediction
        if str(pred_value) in status_mapping:
            return status_mapping[str(pred_value)]
        elif pred_value in [0, 1, 2, 3]:  # If numeric predictions
            status_list = ['Healthy', 'Low Stress', 'Moderate Stress', 'High Stress']
            status_key = status_list[int(pred_value)]
            return status_mapping[status_key]
        else:
            return {
                'status': 'unknown',
                'message': '❓ Unknown Status',
                'description': f'Prediction: {pred_value}. Unable to interpret result.',
                'color_class': 'warning',
                'risk_level': 'Medium',
                'priority': 'Medium'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': '❌ Analysis Error',
            'description': f'Error interpreting prediction: {str(e)}',
            'color_class': 'diseased',
            'risk_level': 'High',
            'priority': 'Critical'
        }

def get_recommendations(status, sensor_data):
    """Get specific recommendations based on status and sensor readings"""
    base_recommendations = {
        'high_stress': [
            'Immediate intervention required',
            'Check watering schedule - soil moisture may be too low/high',
            'Verify environmental conditions (temperature, humidity)',
            'Test soil pH and nutrient levels',
            'Consider relocating plant if environmental stress detected',
            'Monitor plant closely for next 24-48 hours'
        ],
        'moderate_stress': [
            'Adjust watering frequency based on soil moisture readings',
            'Monitor temperature and humidity levels',
            'Check light intensity - may need more/less light',
            'Consider soil pH adjustment if outside optimal range',
            'Review fertilization schedule'
        ],
        'low_stress': [
            'Continue current care routine with minor adjustments',
            'Monitor soil moisture levels daily',
            'Ensure consistent environmental conditions',
            'Regular health checks recommended'
        ],
        'healthy': [
            'Continue current excellent care routine',
            'Maintain consistent watering schedule',
            'Monitor for any changes in sensor readings',
            'Regular preventive care recommended'
        ],
        'unknown': [
            'Monitor all sensor readings closely',
            'Verify sensor calibration',
            'Consult plant care specialist if issues persist'
        ]
    }
    
    # Add specific recommendations based on sensor values
    specific_recs = []
    if isinstance(sensor_data, dict):
        if sensor_data.get('Soil_Moisture', 0) < 20:
            specific_recs.append('⚠️ Soil moisture is low - increase watering frequency')
        elif sensor_data.get('Soil_Moisture', 0) > 80:
            specific_recs.append('⚠️ Soil is too wet - reduce watering to prevent root rot')
            
        if sensor_data.get('Soil_pH', 7) < 6:
            specific_recs.append('🧪 Soil is acidic - consider adding lime to raise pH')
        elif sensor_data.get('Soil_pH', 7) > 8:
            specific_recs.append('🧪 Soil is alkaline - consider adding sulfur to lower pH')
            
        if sensor_data.get('Light_Intensity', 500) < 200:
            specific_recs.append('💡 Light intensity is low - consider moving to brighter location')
            
        if sensor_data.get('Nitrogen_Level', 20) < 10:
            specific_recs.append('🌿 Nitrogen levels are low - consider nitrogen-rich fertilizer')
    
    recommendations = base_recommendations.get(status, base_recommendations['unknown'])
    return recommendations + specific_recs

@app.route('/')
def index():
    """Serve the main page"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plant Health Monitor</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                margin: 0;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            h1 { color: #4CAF50; text-align: center; margin-bottom: 30px; }
            .status-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; 
                margin: 20px 0; 
            }
            .status-card { 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center; 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .loaded { background: #d4edda; color: #155724; }
            .not-loaded { background: #f8d7da; color: #721c24; }
            .endpoint { 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 8px; 
                margin: 10px 0; 
                border-left: 4px solid #007bff;
            }
            .feature-list {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
            }
            .feature-list ul {
                list-style-type: none;
                padding: 0;
            }
            .feature-list li {
                padding: 5px 0;
                border-bottom: 1px solid #dee2e6;
            }
            .feature-list li:last-child {
                border-bottom: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌱 Plant Health Monitor - Backend Server</h1>
            
            <div class="status-grid">
                <div class="status-card {{ 'loaded' if model_loaded else 'not-loaded' }}">
                    <h3>AI Model</h3>
                    <p>{{ 'Loaded ✅' if model_loaded else 'Not Loaded ❌' }}</p>
                </div>
                <div class="status-card {{ 'loaded' if data_loaded else 'not-loaded' }}">
                    <h3>Sample Data</h3>
                    <p>{{ 'Available ✅' if data_loaded else 'Not Found ❌' }}</p>
                </div>
            </div>
            
            <div class="endpoint">
                <h3>🔌 API Endpoints</h3>
                <p><strong>Health Check:</strong> GET /health</p>
                <p><strong>Predict from Sensors:</strong> POST /predict</p>
                <p><strong>Get Sample Data:</strong> GET /sample-data</p>
            </div>
            
            <div class="endpoint">
                <h3>📊 Expected Sensor Data Format</h3>
                <div class="feature-list">
                    <p>The API expects JSON data with these sensor readings:</p>
                    <ul>
                        {% for feature in feature_columns %}
                        <li>{{ loop.index }}. {{ feature }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
            
            <div class="endpoint">
                <h3>💡 Usage Example</h3>
                <p>Send POST request to <code>/predict</code> with JSON payload:</p>
                <pre style="background: #f1f3f4; padding: 10px; border-radius: 5px; font-size: 0.9em;">
{
    "Soil_Moisture": 25.5,
    "Ambient_Temperature": 22.3,
    "Soil_Temperature": 20.1,
    "Humidity": 60.2,
    "Light_Intensity": 450.0,
    "Soil_pH": 6.5,
    "Nitrogen_Level": 15.2,
    "Phosphorus_Level": 25.8,
    "Potassium_Level": 30.1,
    "Chlorophyll_Content": 40.5,
    "Electrochemical_Signal": 0.85
}</pre>
            </div>
        </div>
    </body>
    </html>
    """, 
    model_loaded=(model is not None),
    data_loaded=(data_sample is not None),
    feature_columns=FEATURE_COLUMNS
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Predict plant health from sensor data"""
    start_time = datetime.now()
    
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 500
        
        # Get sensor data from request
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No sensor data provided',
                'success': False
            }), 400
        
        # Validate and preprocess data
        features, error = validate_sensor_data(data)
        if error:
            return jsonify({
                'error': error,
                'success': False
            }), 400
        
        # Make prediction
        try:
            prediction = model.predict(features)
            
            # Get prediction probabilities if available
            try:
                probabilities = model.predict_proba(features)
                confidence = np.max(probabilities) * 100
                class_probs = {
                    f'Class_{i}': float(prob) for i, prob in enumerate(probabilities[0])
                }
            except:
                confidence = 85.0  # Default confidence
                class_probs = {}
            
        except Exception as e:
            return jsonify({
                'error': f'Prediction failed: {str(e)}',
                'success': False
            }), 500
        
        # Interpret results
        status_info = interpret_prediction(prediction, probabilities)
        recommendations = get_recommendations(status_info['status'], data)
        
        # Calculate analysis time
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'raw_prediction': str(prediction[0]) if isinstance(prediction, (list, np.ndarray)) else str(prediction),
                'status': status_info['status'],
                'message': status_info['message'],
                'description': status_info['description'],
                'color_class': status_info['color_class'],
                'confidence': round(confidence, 1),
                'risk_level': status_info['risk_level'],
                'priority': status_info['priority']
            },
            'sensor_analysis': {
                'input_data': data,
                'processed_features': features.tolist()[0],
                'feature_names': FEATURE_COLUMNS
            },
            'probabilities': class_probs,
            'recommendations': recommendations,
            'analysis_time': round(analysis_time, 3),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'success': False
        }), 500

@app.route('/sample-data', methods=['GET'])
def get_sample_data():
    """Get sample sensor data for testing"""
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            # Get the last few records as samples
            samples = df.tail(5)[FEATURE_COLUMNS].to_dict('records')
            
            return jsonify({
                'success': True,
                'samples': samples,
                'feature_columns': FEATURE_COLUMNS,
                'total_records': len(df)
            })
        else:
            # Return dummy data if no CSV available
            dummy_data = {
                'Soil_Moisture': 25.5,
                'Ambient_Temperature': 22.3,
                'Soil_Temperature': 20.1,
                'Humidity': 60.2,
                'Light_Intensity': 450.0,
                'Soil_pH': 6.5,
                'Nitrogen_Level': 15.2,
                'Phosphorus_Level': 25.8,
                'Potassium_Level': 30.1,
                'Chlorophyll_Content': 40.5,
                'Electrochemical_Signal': 0.85
            }
            
            return jsonify({
                'success': True,
                'samples': [dummy_data],
                'feature_columns': FEATURE_COLUMNS,
                'note': 'Using dummy data - CSV file not found'
            })
            
    except Exception as e:
        return jsonify({
            'error': f'Error loading sample data: {str(e)}',
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_available': data_sample is not None,
        'feature_columns': FEATURE_COLUMNS,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Plant Health Monitor Backend...")
    print("📊 Expected input: Sensor data (not images)")
    
    # Load the model and sample data
    model_loaded = load_model()
    data_loaded = load_sample_data()
    
    if model_loaded:
        print("✅ Backend setup complete!")
        print("📡 API will be available at: http://localhost:5000")
        print("🔍 Prediction endpoint: http://localhost:5000/predict")
        print("📊 Sample data endpoint: http://localhost:5000/sample-data")
        print("\n📋 Required sensor data format:")
        for i, col in enumerate(FEATURE_COLUMNS, 1):
            print(f"  {i:2d}. {col}")
    else:
        print("⚠️  Backend starting without model. Please check the model file.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)