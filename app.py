import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class WorkoutRecommendationSystem:
    def __init__(self):
        self.model = None
        self.bmi_status_encoder = None
        self.age_category_encoder = None
        self.workout_encoder = None
        self.workout_decoder = None
        self.scaler = None
        
    def train(self, dataset_path='dataset1.csv'):
        """Train the workout recommendation model"""
        print("Using TensorFlow version:", tf.__version__)

        # Load dataset
        print("Loading and preprocessing data...")
        df = pd.read_csv(dataset_path)

        # Check for missing values
        print("\nMissing values count:")
        print(df.isna().sum())

        # Data preprocessing
        # Replace NaN values with empty string for workout columns
        workout_cols = ['Workout1', 'Workout2', 'Workout3', 'Workout4', 'Workout5']
        df[workout_cols] = df[workout_cols].fillna('')
        
        # Create age category based on age ranges
        df['Kategori_Usia'] = df['Umur'].apply(self.get_age_category)
        
        # BMI status is already in the dataset as BMI_status
        
        # Check BMI distribution
        print("\nBMI status distribution:")
        print(df['BMI_status'].value_counts())
        print("\nAge category distribution:")
        print(df['Kategori_Usia'].value_counts())

        # Create encoders
        self.bmi_status_encoder = LabelEncoder()
        self.age_category_encoder = LabelEncoder()

        # Fit and transform categorical features
        df['BMI_status_Encoded'] = self.bmi_status_encoder.fit_transform(df['BMI_status'])
        df['Kategori_Usia_Encoded'] = self.age_category_encoder.fit_transform(df['Kategori_Usia'])

        # Print encoder mappings
        print("\nBMI Status Mapping:")
        for i, category in enumerate(self.bmi_status_encoder.classes_):
            print(f"{category}: {i}")

        print("\nAge Category Mapping:")
        for i, category in enumerate(self.age_category_encoder.classes_):
            print(f"{category}: {i}")

        # Scale numerical features
        self.scaler = MinMaxScaler()
        numerical_cols = ['tinggi_badan', 'berat_badan', 'Umur']
        # Add BMI calculation for consistency, even though BMI_status exists
        df['BMI'] = df['berat_badan'] / ((df['tinggi_badan'] / 100) ** 2)
        numerical_cols.append('BMI')
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])

        # Create a mapping of workout types
        unique_workouts = set()
        for col in workout_cols:
            unique_workouts.update(df[col].unique())
        unique_workouts.discard('')  # Remove empty string

        print(f"\nTotal unique workout types: {len(unique_workouts)}")
        print("Workout types:", sorted(unique_workouts))

        # Create workout encoder
        self.workout_encoder = {workout: i for i, workout in enumerate(unique_workouts)}
        self.workout_decoder = {i: workout for workout, i in self.workout_encoder.items()}

        # Create feature matrix X and target matrix Y
        X_features = df[['tinggi_badan', 'berat_badan', 'Umur', 'BMI', 
                         'BMI_status_Encoded', 'Kategori_Usia_Encoded']].values

        Y = np.array([self._get_workout_encodings(row, workout_cols) for _, row in df.iterrows()])

        print(f"\nFeature matrix shape: {X_features.shape}")
        print(f"Target matrix shape: {Y.shape}")

        # Print sample count for each workout type
        print("\nSample counts for each workout type:")
        workout_counts = Y.sum(axis=0)
        for i, count in enumerate(workout_counts):
            if count > 0:  # Only show workouts that appear in the dataset
                print(f"{self.workout_decoder[i]}: {int(count)}")

        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.2, random_state=42)

        # Build and train model
        print("\nBuilding and training the model...")
        self.model = self._build_recommendation_model(X_features.shape[1], len(unique_workouts))

        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()

        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
        )

        # Train the model
        history = self.model.fit(
            X_train, Y_train,
            epochs=30,
            batch_size=8,
            validation_data=(X_test, Y_test),
            verbose=1,
            callbacks=[early_stopping, reduce_lr]
        )

        # Evaluate model on test data
        test_loss, test_hamming_loss, test_f1, test_accuracy = self.model.evaluate(X_test, Y_test)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Hamming Loss: {test_hamming_loss:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test Binary Accuracy: {test_accuracy:.4f}")

        # Save model and components
        self.save_model('model')
        
        print("\nDone! The model has been trained successfully.")
        
        # Return metrics for easier access
        return {
            'test_loss': test_loss,
            'test_hamming_loss': test_hamming_loss,
            'test_f1': test_f1,
            'test_accuracy': test_accuracy
        }
    
    def _get_workout_encodings(self, row, workout_cols):
        """Convert row's workout data to one-hot encoding"""
        workout_encoding = np.zeros(len(self.workout_encoder))
        for col in workout_cols:
            if row[col] and row[col] in self.workout_encoder:
                workout_encoding[self.workout_encoder[row[col]]] = 1
        return workout_encoding
    
    def _build_recommendation_model(self, input_shape, num_workouts):
        """Build the neural network model"""
        # Input layer
        inputs = Input(shape=(input_shape,))
        
        # Hidden layers
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(num_workouts, activation='sigmoid')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Define custom metrics for multi-label classification
        def hamming_loss(y_true, y_pred):
            # Convert predictions to binary (0 or 1)
            threshold = 0.5
            y_pred_binary = tf.cast(tf.greater_equal(y_pred, threshold), tf.float32)
            
            # Calculate Hamming loss
            return tf.reduce_mean(tf.cast(tf.not_equal(y_true, y_pred_binary), tf.float32))
        
        def f1_score(y_true, y_pred):
            # Convert predictions to binary (0 or 1)
            threshold = 0.5
            y_pred_binary = tf.cast(tf.greater_equal(y_pred, threshold), tf.float32)
            
            # Calculate true positives, false positives, false negatives
            true_positives = tf.reduce_sum(y_true * y_pred_binary)
            false_positives = tf.reduce_sum((1 - y_true) * y_pred_binary)
            false_negatives = tf.reduce_sum(y_true * (1 - y_pred_binary))
            
            # Calculate precision and recall
            precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
            recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
            
            # Calculate F1 score
            return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[hamming_loss, f1_score, tf.keras.metrics.BinaryAccuracy(threshold=0.3)]
        )
        
        return model
    
    def get_bmi_category(self, bmi):
        """Determine BMI category from BMI value"""
        if bmi < 18.5:
            return 'underweight'
        elif bmi < 25:
            return 'normal'
        elif bmi < 30:
            return 'overweight'
        else:
            return 'obesitas'
    
    def get_age_category(self, age):
        """Determine age category from age value"""
        if age < 36:
            return 'Muda'
        elif age < 61:
            return 'Dewasa'
        else:
            return 'Tua'
    
    def recommend(self, height, weight, age, include_details=False):
        """
        Recommend workouts based on height, weight, and age
        
        Parameters:
        - height: Height in cm
        - weight: Weight in kg
        - age: Age in years
        - include_details: If True, return scores and categories
        
        Returns:
        - If include_details=False: List of recommended workouts, number of recommended workouts
        - If include_details=True: List of (workout, score) tuples, number of workouts, BMI category, age category
        """
        # Calculate BMI
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        
        # Get categories
        bmi_category = self.get_bmi_category(bmi)
        age_category = self.get_age_category(age)
        
        # Encode categories
        try:
            bmi_encoded = self.bmi_status_encoder.transform([bmi_category])[0]
            age_encoded = self.age_category_encoder.transform([age_category])[0]
        except:
            print(f"Warning: Could not encode BMI category '{bmi_category}' or age category '{age_category}'.")
            print("Using default encoding values.")
            bmi_encoded = 0
            age_encoded = 0
        
        # Scale features
        features = np.array([[height, weight, age, bmi]])
        scaled_features = self.scaler.transform(features)
        
        # Create input array
        input_data = np.array([[
            scaled_features[0][0],  # Height
            scaled_features[0][1],  # Weight
            scaled_features[0][2],  # Age
            scaled_features[0][3],  # BMI
            bmi_encoded,            # BMI category
            age_encoded             # Age category
        ]])
        
        # Get predictions
        predictions = self.model.predict(input_data)[0]
        
        # Determine number of workouts based on age category 
        # (following the pattern from the dataset)
        if age_category == 'Muda':
            num_workouts = 5  # Young gets 5 workouts
        elif age_category == 'Dewasa':
            num_workouts = 4  # Adults get 4 workouts
        else:  # Tua (Elderly)
            num_workouts = 3  # Elderly get 3 workouts
        
        # Get top workout indices
        top_indices = predictions.argsort()[-num_workouts:][::-1]
        
        if include_details:
            # Return workouts with confidence scores and categories
            recommendations = [(self.workout_decoder[idx], float(predictions[idx])) for idx in top_indices]
            return recommendations, num_workouts, bmi_category, age_category
        else:
            # Return only workout names
            recommendations = [self.workout_decoder[idx] for idx in top_indices]
            return recommendations, num_workouts
    
    def save_model(self, model_dir):
        """Save the model and all necessary components for later use"""
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save TensorFlow model
        self.model.save(os.path.join(model_dir, 'workout_model.h5'))
        
        # Save encoders and scaler
        with open(os.path.join(model_dir, 'bmi_status_encoder.pkl'), 'wb') as f:
            pickle.dump(self.bmi_status_encoder, f)
            
        with open(os.path.join(model_dir, 'age_category_encoder.pkl'), 'wb') as f:
            pickle.dump(self.age_category_encoder, f)
            
        with open(os.path.join(model_dir, 'workout_encoder.pkl'), 'wb') as f:
            pickle.dump(self.workout_encoder, f)
            
        with open(os.path.join(model_dir, 'workout_decoder.pkl'), 'wb') as f:
            pickle.dump(self.workout_decoder, f)
            
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print(f"Model and components saved to '{model_dir}' directory")
    
    def load_model(self, model_dir):
        """Load the model and all necessary components"""
        # Custom objects for loading the model
        custom_objects = {
            'hamming_loss': self._hamming_loss_fn,
            'f1_score': self._f1_score_fn
        }
        
        # Load TensorFlow model
        self.model = load_model(os.path.join(model_dir, 'workout_model.h5'), custom_objects=custom_objects)
        
        # Load encoders and scaler
        with open(os.path.join(model_dir, 'bmi_status_encoder.pkl'), 'rb') as f:
            self.bmi_status_encoder = pickle.load(f)
            
        with open(os.path.join(model_dir, 'age_category_encoder.pkl'), 'rb') as f:
            self.age_category_encoder = pickle.load(f)
            
        with open(os.path.join(model_dir, 'workout_encoder.pkl'), 'rb') as f:
            self.workout_encoder = pickle.load(f)
            
        with open(os.path.join(model_dir, 'workout_decoder.pkl'), 'rb') as f:
            self.workout_decoder = pickle.load(f)
            
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
            
        print(f"Model and components loaded from '{model_dir}' directory")
    
    def _hamming_loss_fn(self, y_true, y_pred):
        """Hamming loss function for model loading"""
        # Convert predictions to binary (0 or 1)
        threshold = 0.5
        y_pred_binary = tf.cast(tf.greater_equal(y_pred, threshold), tf.float32)
        
        # Calculate Hamming loss
        return tf.reduce_mean(tf.cast(tf.not_equal(y_true, y_pred_binary), tf.float32))
    
    def _f1_score_fn(self, y_true, y_pred):
        """F1 score function for model loading"""
        # Convert predictions to binary (0 or 1)
        threshold = 0.5
        y_pred_binary = tf.cast(tf.greater_equal(y_pred, threshold), tf.float32)
        
        # Calculate true positives, false positives, false negatives
        true_positives = tf.reduce_sum(y_true * y_pred_binary)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred_binary)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred_binary))
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
        recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
        
        # Calculate F1 score
        return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())


# Initialize the system
system = WorkoutRecommendationSystem()


# Add a simple index route for the root path
@app.route('/', methods=['GET'])
def index():
    return """
   <html>
<head>
    <title>Workout Recommendation API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        code {
            background-color: #f0f0f0;
            padding: 2px 5px;
            border-radius: 3px;
        }
        pre {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .endpoint {
            margin-bottom: 20px;
        }
        .endpoint h2 {
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <h1>Workout Recommendation API</h1>
    <p>Welcome to the Workout Recommendation API. Use the following endpoints:</p>
    
    <div class="endpoint">
        <h2>Health Check</h2>
        <code>GET /api/health</code>
        <p>Returns the status of the API</p>
    </div>
    
    <div class="endpoint">
        <h2>Workout Recommendation</h2>
        <code>POST /api/recommend</code>
        <p>Get personalized workout recommendations based on BMI, height, weight, and age</p>
        <h3>Request Body:</h3>
        <pre>
{
  "age": 30,
  "height": 175,
  "weight": 70
}
        </pre>
        <h3>Response:</h3>
        <pre>
{
  "data": {
    "age": 30,
    "age_category": "Muda",
    "bmi": 22.86,
    "bmi_category": "normal",
    "height": 175,
    "recommended_workouts": [
      {
        "confidence": 0.95,
        "name": "pull up"
      },
      {
        "confidence": 0.88,
        "name": "push up"
      },
      {
        "confidence": 0.67,
        "name": "jogging"
      }
    ],
    "weight": 70
  },
  "success": true
}
        </pre>
    </div>
</body>
</html>
    """


@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        # Get dataset path from request (optional)
        request_data = request.get_json(silent=True) or {}
        dataset_path = request_data.get('dataset_path', 'dataset1.csv')
        
        # Train the model
        metrics = system.train(dataset_path=dataset_path)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error training model: {str(e)}'
        }), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_workouts():
    try:
        # Load model if not already loaded
        if system.model is None:
            try:
                system.load_model('model')
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'Model not found. Please train the model first: {str(e)}'
                }), 404
        
        # Get parameters from request
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        # Extract parameters
        height = float(data.get('height', 0))
        weight = float(data.get('weight', 0))
        age = int(data.get('age', 0))
        
        # Validate parameters
        if height <= 0 or weight <= 0 or age <= 0:
            return jsonify({
                'success': False,
                'message': 'Invalid parameters. Height, weight and age must be positive values.'
            }), 400
        
        # Get recommendations with details
        recommendations, num_workouts, bmi_cat, age_cat = system.recommend(height, weight, age, include_details=True)
        
        # Calculate BMI
        bmi_value = weight / ((height/100)**2)
        
        # Prepare response
        response = {
            'success': True,
            'data': {
                'height': height,
                'weight': weight,
                'age': age,
                'bmi': round(bmi_value, 2),
                'bmi_category': bmi_cat,
                'age_category': age_cat,
                'recommended_workouts': [
                    {'name': workout, 'confidence': score} for workout, score in recommendations
                ]
            }
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error recommending workouts: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Workout Recommendation API is running'
    })

if __name__ == "__main__":
    # Get port from environment variable (for Railway)
    port = int(os.environ.get("PORT", 5000))
    
    # Check if model exists, if not train it
    if not os.path.exists('model/workout_model.h5'):
        print("No model found. Training a new model...")
        system.train()
    else:
        print("Loading existing model...")
        system.load_model('model')
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port)
