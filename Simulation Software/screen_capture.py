import mss
import pyautogui as pag
import cv2
import numpy as np
import time


class ScreenCapture:

    def __init__(self, bbox=(0, 44, 1000, 780)):

        self.bbox = bbox
        image = self.capture()
        self.height, self.width = image.shape[:2]

    def start(self):

        self.__swipe()

    @staticmethod
    def __swipe():

        pag.keyDown('ctrl')
        pag.press('left')
        pag.keyUp('ctrl')
        print('Swipe done')
        time.sleep(1)
        print('Pausing')

        cv2.namedWindow('feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('feed', 400, 300)
        cv2.moveWindow('feed', 1020, 100)

    @staticmethod
    def display(im, elapsed_time=0.0):

        # Write fps on image before displaying if elapsed time returned
        if elapsed_time != 0.0:
            fps = 1 / elapsed_time
            cv2.putText(im, 'FPS is: ' + str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

        # Create window and show image
        cv2.imshow('feed', im)

    def capture(self, timed=False, display=False):

        # Start timer
        start_time = time.time()

        # Take screenshot
        with mss.mss() as sct:
            im = sct.grab(self.bbox)

        # Convert image to numpy array for OpenCV
        im = np.array(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

        # If timing end timer calculate time and return image and time
        if timed:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if display:
                self.display(im, elapsed_time=elapsed_time)

            return im, elapsed_time
        else:
            if display:
                self.display(im)

            return im


