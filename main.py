from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the trained model
model = load_model('traffic_sign_model.keras')

# Define class descriptions
class_descriptions = [
    "Speed Limit 20 km/h", "Speed Limit 30 km/h", "Speed Limit 50 km/h",
    "Speed Limit 60 km/h", "Speed Limit 70 km/h", "Speed Limit 80 km/h",
    "End of Speed Limit", "Speed Limit 100 km/h", "Speed Limit 120 km/h",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road",
    "Yield", "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
    "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing by vechiles over 3.5 metric tons"
]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        if file:
            try:
                # Load and preprocess the image
                image = Image.open(file)
                image = image.resize((30, 30))
                image = np.array(image)
                image = image / 255.0
                image = np.expand_dims(image, axis=0)

                # Make a prediction
                prediction = model.predict(image)
                predicted_class = np.argmax(prediction)

                # Get the class description
                predicted_label = class_descriptions[predicted_class]


                # Display the image and prediction
                plt.figure(figsize=(5, 5))
                plt.imshow(image[0])
                plt.title(f"Predicted class: {class_descriptions[predicted_class]}")
                plt.axis('off')
                plt.show()

                return render_template('index.html', prediction=predicted_label)
            except Exception as e:
                return f"Error: {str(e)}"
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
