import cv2
import numpy as np
x = np.random.randint(0,255,(100,100,3)).astype('uint8')
writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 25, (100,100), True)
for k in range(100):
  writer.write(x)
writer.release()
