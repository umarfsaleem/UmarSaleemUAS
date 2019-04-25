import pyautogui as pag
import numpy as np
import cv2
import time
import mss
import pyscreenshot as psc

"""
This script is used to test the speed of a number of screenshot methods. 
It aims to take a screenshot of an isolated area by either taking a screenshot
of the whole screen and cropping afterwards or taking the screenshot only of a 
specific area

The code is not massively well written but it was just for testing so this was 
mostly unimportant
"""

pag.keyDown('ctrl')
pag.press('left')
pag.keyUp('ctrl')
print('Swipe done')
time.sleep(1)
print('Pausing')

images = []

# mss post grab crop
print('Grabbing')
start_time = time.time()

with mss.mss() as sct:
    im = sct.grab(sct.monitors[0])
    im = np.array(im)
    im_crop = im[0:1500, 0:2000]

end_time = time.time()
elapsed_time = end_time - start_time
print('Grabbed')
print('MSS post cropped screenshot took: ' + str(elapsed_time) + ' seconds to complete')

images.append(im_crop)

# mss pre grab crop
print('Grabbing')
start_time = time.time()

with mss.mss() as sct:
    im = sct.grab((0, 44, 1000, 770))
    im = np.array(im)

end_time = time.time()
elapsed_time = end_time - start_time
print('Grabbed')
print('MSS pre cropped screenshot took: ' + str(elapsed_time) + ' seconds to complete')

images.append(im)

# pag post cropped
print('Grabbing')
start_time = time.time()

im = pag.screenshot()
im_np = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
im_crop = im_np[0:1500, 0:2000]

end_time = time.time()
elapsed_time = end_time - start_time
print('Grabbed')
print('Pyautogui post cropped screenshot took: ' + str(elapsed_time) + ' seconds to complete')

images.append(im_crop)

# pag pre cropped
print('Grabbing')
start_time = time.time()

im = pag.screenshot(region=(0, 0, 2000, 1500))
im_np = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

end_time = time.time()
elapsed_time = end_time - start_time
print('Grabbed')
print('Pyautogui pre cropped screenshot took: ' + str(elapsed_time) + ' seconds to complete')

images.append(im_np)

# psc post cropped
print('Grabbing')
start_time = time.time()

im = psc.grab(backend='mac_quartz', childprocess=False)
im_crop = im_np[0:1500, 0:2000]
im_np = np.array(im_crop)

end_time = time.time()
elapsed_time = end_time - start_time
print('Grabbed')
print('Pyscreenshot post cropped screenshot took: ' + str(elapsed_time) + ' seconds to complete')

images.append(im_np)

# psc pre cropped
print('Grabbing')
start_time = time.time()

im = psc.grab(backend='mac_quartz', childprocess=False, bbox=(0, 44, 1000, 770))
im_np = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

end_time = time.time()
elapsed_time = end_time - start_time
print('Grabbed')
print('Pyscreenshot pre cropped screenshot took: ' + str(elapsed_time) + ' seconds to complete')

images.append(im_np)

count = 0
for image in images:
    count = count + 1
    cv2.namedWindow("image " + str(count), cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image " + str(count), 600, 400)
    cv2.imshow("image " + str(count), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


