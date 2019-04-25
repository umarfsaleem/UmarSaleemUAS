import pyautogui as pag
import numpy as np
import cv2
import time
import pyscreenshot as psc

pag.keyDown('ctrl')
pag.press('left')
pag.keyUp('ctrl')
print('Swipe done')
time.sleep(1)
print('Pausing')

print('Grabbing')
start_time = time.time()
im = psc.grab(backend='mac_quartz', childprocess=False)
end_time = time.time()
elapsed_time = end_time - start_time
print('Grabbed')
print('Screenshot took: ' + str(elapsed_time) + ' seconds to complete')

im_numpy = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 400)
cv2.imshow('image', im_numpy)
print('Displaying done')
cv2.waitKey(0)
cv2.destroyAllWindows()
