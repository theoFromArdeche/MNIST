import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Read the CSV file
df = pd.read_csv('sign_mnist_test.csv')

# Set the directory to save the images
folder_save = 'dataset_signs_test'
for k in range(25):
  if (k==9): continue
  if not os.path.exists(folder_save + '/' + str(k)):
    os.makedirs(folder_save + '/' + str(k))

# Set up a matplotlib figure and axis for displaying the images
fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# Loop through the rows of the dataframe
print(df.shape)
for i, row in df.iterrows():
  print(i)
  # Extract the label and the pixel values from the row
  label = row['label']
  pixel_values = row.drop('label').values.flatten()

  # Reshape the pixel values into a 28x28 array
  image_array = np.reshape(pixel_values, (28, 28))

  # Display the image
  plt.imshow(image_array, cmap=plt.cm.binary)

  # Save the image to the appropriate subdirectory
  plt.savefig(folder_save + '/' + str(label) + '/' + str(i) + '.jpeg', dpi=10)

  # Clear the current axis and figure
  plt.cla()
  plt.clf()
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
