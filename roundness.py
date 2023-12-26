import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('image_measurements.csv')  # Replace 'your_grains_data.csv' with your actual file path

# Calculate roundness and add it as a new column
df['Roundness'] = df['MajorAxisLength'] / df['MinorAxisLength']

# Define roundness categories
very_round = (0.9, 1.1)  # Adjust the range as needed
round_category = (0.8, 0.9)
irregular = (0.6, 0.8)
highly_irregular = (0, 0.6)

# Categorize roundness
df['RoundnessCategory'] = pd.cut(df['Roundness'], bins=[very_round[0], round_category[0], irregular[0], highly_irregular[0], 1], labels=['Very Round', 'Round', 'Irregular', 'Highly Irregular'])

# Plot histogram
plt.figure(figsize=(10, 6))
df['RoundnessCategory'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Roundness Categories Histogram')
plt.xlabel('Roundness Category')
plt.ylabel('Count')
plt.show()
