import pyautogui as pag
import numpy as np
import cv2
import time
import mss


pag.keyDown('ctrl')
pag.press('left')
pag.keyUp('ctrl')
print('Swipe done')
time.sleep(1)
print('Pausing')

print('Grabbing')
start_time = time.time()
with mss.mss() as sct:
    im = sct.grab(sct.monitors[0])
end_time = time.time()
elapsed_time = end_time - start_time
print('Grabbed')
print('Screenshot took: ' + str(elapsed_time) + ' seconds to complete')

print('Grabbing')
start_time = time.time()
with mss.mss() as sct:
    im2 = sct.grab((0, 44, 1000, 770))
end_time = time.time()
elapsed_time = end_time - start_time
print('Grabbed')
print('Cropped screenshot took: ' + str(elapsed_time) + ' seconds to complete')

im_numpy = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
im_nr = np.array(im)
im2 = np.array(im2)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 400)
cv2.imshow('image', im_numpy)
cv2.namedWindow('image non converted', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image non converted', 600, 400)
cv2.imshow('image non converted', im_nr)
cv2.namedWindow('im2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('im2', 600, 400)
cv2.imshow('im2', im2)
print('Displaying done')
cv2.waitKey(0)
cv2.destroyAllWindows()
