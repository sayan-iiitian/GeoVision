import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io  # Add this line for the measure module
import pandas as pd
import numpy as np


# Load the CSV file into a DataFrame
df = pd.read_csv('Photos/image_measurements2.csv')



# Calculate compactness parameters
df['AspectRatio'] = df['MajorAxisLength'] / df['MinorAxisLength']
df['Circularity'] = (4 * np.pi * df['Area']) / (df['Perimeter']**2)
df['Elongation'] = df['MajorAxisLength'] / df['MinorAxisLength']



# Print or use compactness parameters as needed
print(f"Mean Aspect Ratio: {df['AspectRatio'].mean()}")
print(f"Mean Circularity: {df['Circularity'].mean()}")
print(f"Mean Elongation: {df['Elongation'].mean()}")

