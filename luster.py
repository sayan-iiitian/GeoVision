import cv2
import numpy as np
from skimage import feature
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Function to extract LBP features from an image
def extract_lbp_features(image):
    radius = 3
    n_points = 8 * radius
    lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize the histogram
    return hist

# Function to load and preprocess a single image
def load_and_preprocess_single_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    else:
        raise ValueError("Image not loaded successfully.")

# Function to predict luster class for a single image
def predict_luster_class(model, image):
    lbp_features = extract_lbp_features(image.reshape(-1))  # Reshape to 1D array
    prediction = model.predict([lbp_features])
    return prediction[0]

# Main function for image classification based on luster
def main():
    # Specify the path to your labeled image
    image_path = 'Photos/dull.jpeg'

    # Load and preprocess the single image
    image = load_and_preprocess_single_image(image_path)

    # Load the trained classification model
    trained_model_path = 'path/to/your/trained_model.pkl'  # Specify the path to your trained model
    luster_classifier = joblib.load(trained_model_path)

    # Predict the luster class for the single image
    predicted_class = predict_luster_class(luster_classifier, image)

    # Print the predicted class
    print(f"Predicted Luster Class: {predicted_class}")

if __name__ == "__main__":
    main()
