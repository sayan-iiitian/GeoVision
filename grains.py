from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, color
import csv
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
CSV_FOLDER = 'csv_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER
pixels_to_um = 0.5  # 1 pixel = 500 nm (got this from the metadata of the original image)

propList = [
    'Area',
    'equivalent_diameter',  # Added... verify if it works
    'orientation',  # Added, verify if it works. Angle between x-axis and major axis.
    'MajorAxisLength',
    'MinorAxisLength',
    'Perimeter',
    'MinIntensity',
    'MeanIntensity',
    'MaxIntensity'
]

def process_image_and_save_csv(img_data):
    # Convert the image data to a NumPy array
    nparr = np.frombuffer(img_data.read(), np.uint8)
    img1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Call the grain segmentation function.
    regions = grain_segmentation(img, img1)

    # Generate a unique filename for the CSV file
    csv_filename = f"{str(uuid.uuid4())}.csv"
    csv_file_path = os.path.join(app.config['CSV_FOLDER'], csv_filename)

    # Write CSV header
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['FileName', 'Grain #'] + propList)

        grain_number = 1
        for region_props in regions:
            # Write image filename and grain number
            csv_writer.writerow([csv_filename, grain_number])

            # Write cluster properties to the CSV file
            for i, prop in enumerate(propList):
                if prop == 'Area':
                    to_print = region_props[prop] * pixels_to_um**2  # Convert pixel square to um square
                elif prop == 'orientation':
                    to_print = region_props[prop] * 57.2958  # Convert to degrees from radians
                elif prop.find('Intensity') < 0:  # Any prop without Intensity in its name
                    to_print = region_props[prop] * pixels_to_um
                else:
                    to_print = region_props[prop]  # Remaining props, basically the ones with Intensity in their name
                csv_writer.writerow(['', '', to_print])

            grain_number += 1

    return csv_file_path

def grain_segmentation(img, img1):
#Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
    ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Morphological operations to remove small noise - opening
#To remove holes we can use closing
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)


#Now we know that the regions at the center of cells is for sure cells
#The region far away is background.
#We need to extract sure regions. For that we can use erode. 
#But we have cells touching, so erode alone will not work. 
#To separate touching objects, the best approach would be distance transform and then thresholding.

# let us start by identifying sure background area
# dilating pixes a few times increases cell boundary to background. 
# This way whatever is remaining for sure will be background. 
#The area in between sure background and foreground is our ambiguous area. 
#Watershed should find this area for us. 
    sure_bg = cv2.dilate(opening,kernel,iterations=2)


# Finding sure foreground area using distance transform and thresholding
#intensities of the points inside the foreground regions are changed to 
#distance their respective distances from the closest 0 value (boundary).
#https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)


#Let us threshold the dist transform by starting at 1/2 its max value.
#print(dist_transform.max()) gives about 21.9
    ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)

#0.2* max value seems to separate the cells well.
#High value like 0.5 will not recognize some grain boundaries.

# Unknown ambiguous region is nothing but bkground - foreground
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

#Now we create a marker and label the regions inside. 
# For sure regions, both foreground and background will be labeled with positive numbers.
# Unknown regions will be labeled 0. 
#For markers let us use ConnectedComponents. 
    ret3, markers = cv2.connectedComponents(sure_fg)

#One problem rightnow is that the entire background pixels is given value 0.
#This means watershed considers this region as unknown.
#So let us add 1 to all labels so that sure background is not 0, but 1
    markers = markers+10

# Now, mark the region of unknown with zero
    markers[unknown==255] = 0
#plt.imshow(markers)   #Look at the 3 distinct regions.

#Now we are ready for watershed filling. 
    markers = cv2.watershed(img1,markers)
#The boundary region will be marked -1
#https://docs.opencv.org/3.3.1/d7/d1b/group__imgproc__misc.html#ga3267243e4d3f95165d55a618c65ac6e1

#Let us color boundaries in yellow. 
    img1[markers == -1] = [0,255,255]  

    img2 = color.label2rgb(markers, bg_label=0)

    cv2.imshow('Overlay on original image', img1)
    cv2.imshow('Colored Grains', img2)
    cv2.waitKey(0)

#Now, time to extract properties of detected cells
# regionprops function in skimage measure module calculates useful parameters for each object.
    regions = measure.regionprops(markers, intensity_image=img)
    return regions
   

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    # Process the image and get the path to the generated CSV file
    csv_file_path = process_image_and_save_csv(image_file)

    return jsonify({'csv_file_path': csv_file_path})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CSV_FOLDER'], exist_ok=True)
    app.run(debug=True)
