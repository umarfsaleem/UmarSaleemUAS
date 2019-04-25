import pyautogui as pag
import numpy as np
import cv2
import time


pag.keyDown('ctrl')
pag.press('left')
pag.keyUp('ctrl')
print('Swipe done')
pag.PAUSE = 5
print('Pausing')

start_time = time.time()
im = pag.screenshot(region=(0, 0, 1999, 1651))
end_time = time.time()
elapsed_time = end_time - start_time
print('region screenshot takes: ' + str(elapsed_time) + 'seconds to execute')

start_time = time.time()
im2 = pag.screenshot()
im_numpy2 = cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2BGR)
mask = np.zeros(im_numpy2.shape[:2], dtype='uint8')
cv2.rectangle(mask, (0, 50), (1999, 1490), 255, -1)
im_numpy2 = cv2.bitwise_and(im_numpy2, im_numpy2, mask=mask)
end_time = time.time()
elapsed_time = end_time - start_time
print('full screenshot takes: ' + str(elapsed_time) + 'seconds to execute')
print('Screenshot done')

im_numpy = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
print('Reformatting done')

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('image 2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 400)
cv2.resizeWindow('image 2', 600, 400)
cv2.imshow('image', im_numpy)
cv2.imshow('image 2', im_numpy2)
print('Displaying done')
cv2.waitKey(0)
cv2.destroyAllWindows()

