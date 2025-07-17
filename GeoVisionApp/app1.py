from flask import Flask, request, send_file, jsonify
from flask_cors import CORS  # Import the CORS extension
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
import os
import csv
import matplotlib
matplotlib.use('Agg')
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

    # Save the pie chart to the static folder
    static_folder = 'static'
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    img_path = os.path.join(static_folder, 'pie_chart.png')
    plt.savefig(img_path, format='png')
    plt.close()  # Close the plot to free up resources

    return img_path

@app.route('/colors', methods=['POST'])
def get_colors():
    try:
        # Receive the image from the Flutter app
        img_data = request.files['image'].read()
        img_np = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Process the image and get the BytesIO object containing the pie chart
        processed_img_path = process_image(img)

        # Return the file path to the Flutter app
        return send_file(processed_img_path, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500
def pixel_to_distance(pixel_value, distance_to_object, focal_length):
    return (pixel_value * distance_to_object) / focal_length  # pixel value to real-world distance

@app.route('/fault', methods=['POST'])
def detect_fault():
    try:
        # Get image from POST request
        image_file = request.files['image']
        image_np = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        # Fix focal length to 1
        focal_length = 1

        # Fix distance to 5 meters
        distance_to_object = 5.0

        # Apply GaussianBlur to reduce noise and improve Canny edge detection
        blurred_image = cv2.GaussianBlur(image_np, (5, 5), 0)

        # Apply Canny edge detection to detect edges in the image
        canny_output = cv2.Canny(blurred_image, 80, 150)

        # Apply dilation to connect nearby edges. To make the connecting edges as a single unit edge
        kernel_size_dilation = 5
        kernel_dilation = np.ones((kernel_size_dilation, kernel_size_dilation), np.uint8)
        dilated_image = cv2.dilate(canny_output, kernel_dilation, iterations=1)

        # Apply erosion to remove small edges
        kernel_size_erosion = 5
        kernel_erosion = np.ones((kernel_size_erosion, kernel_size_erosion), np.uint8)
        eroded_image = cv2.erode(dilated_image, kernel_erosion, iterations=1)

        # Find contours in the eroded image
        contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Store fault sizes in a list, Empty list to store the sizes of the detected faults
        fault_sizes = []

        # Calculate and print the size of each detected fault in real-world units
        for i, contour in enumerate(contours):
            fault_size_pixels = cv2.contourArea(contour)
            fault_size_meters = pixel_to_distance(fault_size_pixels, distance_to_object, focal_length)
            fault_sizes.append(fault_size_meters)

        # Write fault sizes to a CSV file
        csv_filename = 'fault_sizes.csv'
        csv_path = os.path.join('static', csv_filename)
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Fault', 'Size (meters)'])
            for i, size in enumerate(fault_sizes):
                csv_writer.writerow([f"Fault {i + 1}", size])

        # Draw contours on the original image
        result_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

        # Save the result image
        result_image_path = os.path.join('static', 'result_image.jpg')
        cv2.imwrite(result_image_path, result_image)

        return jsonify({
            'csv_file': csv_path,
            'result_image': result_image_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)
