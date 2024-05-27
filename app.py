import tempfile
from flask import Flask, request, jsonify, send_file
import os
import cv2
import matplotlib
import numpy as np
from scipy import ndimage
from skimage import measure, color
import pandas as pd
import csv
import uuid
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.express as px

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
CSV_FOLDER = 'csv_files'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

pixels_to_um = 0.5

propList = [
    'Area',
    'equivalent_diameter',
    'orientation',
    'MajorAxisLength',
    'MinorAxisLength',
    'Perimeter',
    'MinIntensity',
    'MeanIntensity',
    'MaxIntensity'
]

latest_csv_content = None

def process_image_and_save_csv(img_data):
    global latest_csv_content

    nparr = np.frombuffer(img_data.read(), np.uint8)
    img1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    regions = grain_segmentation(img, img1)

    csv_filename = f"{str(uuid.uuid4())}.csv"
    csv_file_path = os.path.join(app.config['CSV_FOLDER'], csv_filename)

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header_row = ['FileName', 'Grain #'] + propList
        csv_writer.writerow(header_row)

        for grain_number, region_props in enumerate(regions, start=1):
            row_data = [csv_filename, grain_number]

            for prop in propList:
                if prop == 'Area':
                    to_print = region_props[prop] * pixels_to_um**2
                elif prop == 'orientation':
                    to_print = region_props[prop] * 57.2958
                elif prop.find('Intensity') < 0:
                    to_print = region_props[prop] * pixels_to_um
                else:
                    to_print = region_props[prop]

                row_data.append(to_print)

            csv_writer.writerow(row_data)

    df = pd.read_csv(csv_file_path)
    json_content = df.to_json(orient='records')
    latest_csv_content = json_content

    return csv_file_path
def calculate_compactness(area, perimeter):
    # Implement your compactness calculation logic here
    # Compactness = (perimeter^2) / (4 * pi * area)
    compactness = (perimeter ** 2) / (4 * 3.141592653589793 * area)
    return compactness

@app.route('/compactness', methods=['GET'])
def compactness_analysis():
    global latest_csv_content

    if latest_csv_content is None:
        return jsonify({'error': 'No CSV file has been processed yet'}), 400

    df = pd.read_json(latest_csv_content, orient='records')

    if 'Area' not in df.columns or 'Perimeter' not in df.columns:
        return jsonify({'error': 'Required columns (Area and Perimeter) not found in the CSV file'}), 400

    df['Compactness'] = df.apply(lambda row: calculate_compactness(row['Area'], row['Perimeter']), axis=1)

    compactness_results = {
        'data': df[['Grain #', 'Compactness']].to_dict(orient='records')  # Modify as needed
    }

    return jsonify(compactness_results)
def grain_segmentation(img, img1):
    ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 10
    markers[unknown == 255] = 0

    markers = cv2.watershed(img1, markers)
    img1[markers == -1] = [0, 255, 255]

    img2 = color.label2rgb(markers, bg_label=0)

    cv2.imshow('Overlay on original image', img1)
    cv2.imshow('Colored Grains', img2)
    cv2.waitKey(0)

    regions = measure.regionprops(markers, intensity_image=img)
    return regions

@app.route('/grain_char', methods=['POST'])
def process_image():
    global latest_csv_content

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    csv_file_path = process_image_and_save_csv(image_file)
    return jsonify({'csv_content': latest_csv_content})

@app.route('/sorting', methods=['GET'])
def sorting_analysis():
    global latest_csv_content

    if latest_csv_content is None:
        return jsonify({'error': 'No CSV file has been processed yet'}), 400

    df = pd.read_json(latest_csv_content, orient='records')

    sorting_index_area = np.std(df['Area'])
    sorting_index_diameter = np.std(df['equivalent_diameter'])
    
    # Check if 'orientation' column exists in the DataFrame
    if 'orientation' in df.columns:
        skewness_orientation = df['orientation'].skew()
    else:
        skewness_orientation = None

    kurtosis_major_axis = df['MajorAxisLength'].kurtosis()
    gsd_diameter = np.std(df['equivalent_diameter'])
    sorting_coefficient_area = (df['Area'].quantile(0.84) - df['Area'].quantile(0.16)) / (2 * np.median(df['Area']))

    df['EquivalentDiameter (um)'] = df['equivalent_diameter'] * pixels_to_um
    df['Orientation (degrees)'] = df['orientation'] * 57.2958

    sorting_results = {
        'sorting_index_area': sorting_index_area,
        'sorting_index_diameter': sorting_index_diameter,
        'skewness_orientation': skewness_orientation,
        'kurtosis_major_axis': kurtosis_major_axis,
        'gsd_diameter': gsd_diameter,
        'sorting_coefficient_area': sorting_coefficient_area
    }

    return jsonify(sorting_results)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CSV_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
