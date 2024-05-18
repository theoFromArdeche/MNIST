import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import gzip
import os
imagefile = 'train-images.idx3-ubyte'
imagearray = idx2numpy.convert_from_file(imagefile)
f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8)

folder_save = 'dataset_handwritting'
for k in range(10):
  if not os.path.exists(folder_save + '/' + str(k)):
    os.makedirs(folder_save + '/' + str(k))

fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
length = len(imagearray)
print(length)
#buf = f.read(10000)
for k in range(10000):
  print(k)
  plt.imshow(imagearray[k], cmap=plt.cm.binary)
  buf = f.read(1)
  labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)[0]
  #print(labels)
  #plt.title(labels)

  plt.savefig(folder_save + '/' + str(labels) + '/' + str(k) + '.jpeg', dpi=10)
  plt.cla()
  plt.clf()
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  


