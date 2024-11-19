from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Define the Flask application
app = Flask(__name__)

# Define paths and configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('cnn.h5')

# Define image size and class labels
IMAGE_SIZE = (350, 350)
class_labels = ['large cell carcinoma', 'adenocarcinoma', 'normal', 'squamous cell carcinoma']

# Mapping predicted labels to recommended medicines
medicine_recommendations = {
    'large cell carcinoma': 'Chemotherapy: This is often a primary treatment for large cell carcinoma. Chemotherapy drugs can be given intravenously or orally to kill cancer cells throughout the body.,Targeted Therapy: If specific genetic mutations are identified in the cancer cells, targeted therapy drugs may be used to block the growth and spread of cancer cells.',
    'adenocarcinoma': 'Targeted Therapy: Adenocarcinoma often harbors specific genetic mutations, such as EGFR, ALK, ROS1, or BRAF mutations. Targeted therapy drugs are designed to target these mutations and block the growth and spread of cancer cells.',
    'normal': 'No specific medicine needed',
    'squamous cell carcinoma': 'Radiation Therapy: High-energy radiation beams are used to destroy cancer cells. Radiation therapy can be used alone, with chemotherapy (chemoradiation), or after surgery to reduce the risk of recurrence.'
}

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load and preprocess an image
def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

# Define the home route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def ndex():
    return render_template('about.html')

@app.route('/doctor')
def dex():
    return render_template('doctor.html')    

@app.route('/departments')
def inex():
    return render_template('departments.html')     

@app.route('/doctors')
def nex():
    return render_template('doctors.html')    

# Define the upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image and make predictions
        img = load_and_preprocess_image(file_path, IMAGE_SIZE)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]

        # Get medicine recommendation based on predicted label
        recommended_medicine = medicine_recommendations.get(predicted_label, 'No specific medicine recommendation')

        return render_template('doctors.html', filename=filename, label=predicted_label, medicine=recommended_medicine)

    return redirect(request.url)

# Route to display the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
