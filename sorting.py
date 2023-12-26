import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io  # Add this line for the measure module
import pandas as pd
import numpy as np

pixels_to_um = 0.5 
# Load the CSV file into a DataFrame
df = pd.read_csv('Photos/image_measurements2.csv')

img1= cv2.imread("Photos/rock1.jpg")
    
img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# Additional parameters for sorting
df['EquivalentDiameter (um)'] = df['equivalent_diameter'] * pixels_to_um
df['Orientation (degrees)'] = df['orientation'] * 57.2958

# Sorting Index (σ) based on Area
sorting_index_area = np.std(df['Area'])
print(f"Sorting Index (σ) based on Area: {sorting_index_area}")

# Sorting Index (σ) based on Equivalent Diameter
sorting_index_diameter = np.std(df['EquivalentDiameter (um)'])
print(f"Sorting Index (σ) based on Equivalent Diameter: {sorting_index_diameter}")

# Skewness (S) based on Orientation
skewness_orientation = df['Orientation (degrees)'].skew()
print(f"Skewness (S) based on Orientation: {skewness_orientation}")

# Kurtosis (K) based on MajorAxisLength
kurtosis_major_axis = df['MajorAxisLength'].kurtosis()
print(f"Kurtosis (K) based on MajorAxisLength: {kurtosis_major_axis}")

# Graphic Standard Deviation (GSD) based on Equivalent Diameter
gsd_diameter = np.std(df['EquivalentDiameter (um)'])
print(f"Graphic Standard Deviation (GSD) based on Equivalent Diameter: {gsd_diameter}")

# Sorting Coefficient (Kz) based on Area
sorting_coefficient_area = (df['Area'].quantile(0.84) - df['Area'].quantile(0.16)) / (2 * np.median(df['Area']))
print(f"Sorting Coefficient (Kz) based on Area: {sorting_coefficient_area}")

df['EquivalentDiameter (um)'] = df['equivalent_diameter'] * pixels_to_um
df['Orientation (degrees)'] = df['orientation'] * 57.2958

