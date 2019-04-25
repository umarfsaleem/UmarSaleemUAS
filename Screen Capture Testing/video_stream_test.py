import pyautogui as pag
import numpy as np
import cv2
import time
import mss

total_fps = []


def screen_capture_method(option):
    start_time = time.time()
    if option == 1:
        im = pag.screenshot()
        im_np = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        im_crop = im_np[50:1490, 0:1999]
        # mask = np.zeros(im_np.shape[:2], dtype='uint8')
        # cv2.rectangle(mask, (0, 50), (1999, 1490), 255, -1)
        # im_np = cv2.bitwise_and(im_np, im_np, mask=mask)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return im_crop, elapsed_time
    elif option == 2:
        with mss.mss() as sct:
            im = sct.grab(sct.monitors[0])
        im = np.array(im)
        im_crop = im[100:1550, 0:1999]
        end_time = time.time()
        elapsed_time = end_time - start_time
        return im_crop, elapsed_time
    elif option == 3:
        with mss.mss() as sct:
            im = sct.grab((0, 44, 1000, 770))
        im = np.array(im)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return im, elapsed_time


while True:
    # Change number  in screen_capture_method to change which screenshot api to use
    image, time_taken = screen_capture_method(3)
    fps = 1/time_taken
    total_fps.append(fps)
    cv2.namedWindow('feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('feed', 400, 300)
    cv2.putText(image, 'FPS is: ' + str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
    cv2.imshow('feed', image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        average_fps = sum(total_fps) / len(total_fps)
        print('average fps is: ' + str(average_fps))
        cv2.destroyAllWindows()
        break
