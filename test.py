from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

input_matrix = np.array(Image.open('input.jpg'))
canal = input_matrix.shape[2]
height, width, channel = input_matrix.shape # size of picture

# our filter
kernel = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]], np.float32)
kernel_height, kernel_width = kernel.shape
#bilding output
count = 0
output_matrix = np.zeros((height - kernel_height + 1, width - kernel_width + 1, channel), dtype=np.float32)
output_array = output_matrix.reshape(( (height - kernel_height + 1)*( width - kernel_width + 1)), channel)
#fill output_matrix
for row in range(height - kernel_height + 1):
  for column in range(width - kernel_width + 1):
    for c in range (channel):
      summe = input_matrix[row : row + kernel_height, column : column + kernel_width, c]
      output_array[count, c] = np.sum(summe * kernel)
    count += 1
output_matrix = output_array.reshape(height - kernel_height + 1, width - kernel_width + 1, channel)

# show image
plt.imshow(output_matrix.astype('uint8'))
plt.show()




#and here is more easier way :)
#from PIL import Image, ImageFilter

#input_image = Image.open('input.png')
#pixel_map = input_image.load()

#pixels = list(im.getdata())
#width, height = im.size
#pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
#output_image = input_image.filter(ImageFilter.GaussianBlur(radius = 5))

#output_image.save('output.png')
