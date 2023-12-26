from flask import Flask, request, send_file
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import io

app = Flask(__name__)

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += "{:02x}".format(i)
    return hex_color

def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (900, 600), interpolation=cv2.INTER_AREA)
    img = img.reshape(img.shape[0] * img.shape[1], 3)

    clf = KMeans(n_clusters=5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_

    counts = Counter(color_labels)

    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]

    plt.figure(figsize=(12, 8))
    plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
    
    # Save the pie chart to a BytesIO object
    img_bytesio = io.BytesIO()
    plt.savefig(img_bytesio, format='png')
    img_bytesio.seek(0)
    
    plt.close()  # Close the plot to free up resources
    return img_bytesio

@app.route('/colors', methods=['POST'])
def get_colors():
    try:
        # Receive the image from the Flutter app
        img_data = request.files['image'].read()
        img_np = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Process the image and get the BytesIO object containing the pie chart
        processed_img = process_image(img)

        # Return the pie chart as a file to the Flutter app
        return send_file(processed_img, mimetype='image/png')

    except Exception as e:
        return str(e)


from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import cv2
import numpy as np
from keras.models import load_model
import joblib  # Import joblib for loading RF model

app = Flask(__name__)
api = Api(app)

# Load the trained models and label encoder
cnn_model = load_model('cnn_classifier.h5')  # Replace with the path to your CNN model
rf_model = joblib.load('rf_classifier.joblib')   # Replace with the path to your RF model

# Load the existing label encoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('your_existing_label_encoder.npy')  # Replace with the path to your existing label encoder file

SIZE = 128

class CementingMaterial(Resource):
    def post(self):
        # Get the image file from the request
        file = request.files['image']
        
        # Read and preprocess the image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0) / 255.0  # Normalize pixel values

        # Predict using the neural network
        nn_prediction = cnn_model.predict(img)
        nn_prediction = np.argmax(nn_prediction, axis=-1)
        nn_prediction = label_encoder.inverse_transform(nn_prediction)[0]

        # Predict using the Random Forest model
        rf_prediction = rf_model.predict(img)[0]
        rf_prediction = label_encoder.inverse_transform(rf_prediction)

        # Return the predictions
        result = {
            "neural_network_prediction": nn_prediction,
            "random_forest_prediction": rf_prediction
        }

        return jsonify(result)

# Add the resource to the API with the specified endpoint
api.add_resource(CementingMaterial, '/cementing_material')

class Luster(Resource):
    def post(self):
        # Get the image file from the request
        file = request.files['image']
        
        # Read and preprocess the image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0) / 255.0  # Normalize pixel values

        # Assuming you have a luster CNN model and label encoder loaded
        luster_cnn_model = load_model('your_luster_cnn_model.h5')  # Replace with the path to your luster CNN model
        luster_label_encoder = LabelEncoder()
        luster_label_encoder.classes_ = np.load('your_luster_label_encoder.npy')  # Replace with the path to your luster label encoder file

        # Predict using the luster neural network
        luster_nn_prediction = luster_cnn_model.predict(img)
        luster_nn_prediction = np.argmax(luster_nn_prediction, axis=-1)
        luster_nn_prediction = luster_label_encoder.inverse_transform(luster_nn_prediction)[0]

        # Return the luster prediction
        result = {
            "luster_neural_network_prediction": luster_nn_prediction
        }

        return jsonify(result)

# Add the resource for luster to the API with the specified endpoint
api.add_resource(Luster, '/luster')

if __name__ == '__main__':
    app.run(debug=True)

