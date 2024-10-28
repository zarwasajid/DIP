import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
import random
import matplotlib.pyplot as plt

# Define the directory path and class names
directory = r'C:\Users\HP EliteBook\PycharmProjects\pythonProject6\classes'
classes = ['buses', 'cars', 'bikes']

# Function to extract LBP features
def extract_lbp_features(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate LBP
    lbp = local_binary_pattern(gray, 8, 3, 'uniform')

    # Calculate the histogram of LBP
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))

    # Normalize the histogram
    hist = hist / (hist.sum() + 1e-7)

    return hist

# Load an image
image = cv2.imread('d.jpg')

# Extract LBP features from the input image
input_features = extract_lbp_features(image)

# Randomly select an image from a class
class_name = random.choice(classes)
image_path = os.path.join(directory, class_name, random.choice(os.listdir(os.path.join(directory, class_name))))

# Load the randomly selected image
random_image = cv2.imread(image_path)

# Extract LBP features from the randomly selected image
random_features = extract_lbp_features(random_image)


# Calculate the variance of the LBP histograms
input_variance = np.var(input_features)
random_variance = np.var(random_features)
difference = abs(input_variance - random_variance)
print(f"Difference: {difference}")

# Determine if the images match based on variance
variance_threshold = 0.0002  # adjust this value based on your requirements
match_message_variance = "Images match" if difference < variance_threshold else "Images do not match"

# Plotting histograms and displaying the match message
plt.figure(figsize=(12, 10))

# Display input image
plt.subplot(2, 2, 1)
plt.title('Input Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Display randomly selected image
plt.subplot(2, 2, 2)
plt.title('Random Image')
plt.imshow(cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Plot histogram for input image
plt.subplot(2, 2, 3)
plt.title('Input Image LBP Histogram')
plt.bar(range(256), input_features, color='blue', alpha=0.7)
plt.xlim([0, 256])

# Plot histogram for randomly selected image
plt.subplot(2, 2, 4)
plt.title('Random Image LBP Histogram')
plt.bar(range(256), random_features, color='orange', alpha=0.7)
plt.xlim([0, 256])

# Plot the input image features


# Display the match message
plt.suptitle(f"Input Variance: {input_variance:.4f}, Random Variance: {random_variance:.4f}\n"
             f"Variance Match: {match_message_variance}", fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()