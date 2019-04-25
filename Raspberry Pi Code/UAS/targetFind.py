import numpy as np
import cv2
import datetime
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import math


class TargetFinder:

    def __init__(self, record_video=False):
        print ('target finder started')

        self.boundaries = [([0, 128, 60], [10, 255, 255]),
                           ([165, 128, 60], [179, 255, 255])]

        self.camera = None
        self.rawCapture = None
        self.setup_camera()

        initial_frame = self.capture_frame()
        self.width, self.height = initial_frame.shape[:2]

        self.record = record_video

        if self.record:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            current_time = datetime.datetime.now().strftime("%d-%m-%y-%H:%M")
            output_file = '/home/pi/UAS/FlightVideos/' + current_time + '.avi'
            self.out = cv2.VideoWriter(output_file, fourcc, 7, (720, 480))

    def setup_camera(self):
        self.camera = PiCamera()
        self.camera.vflip = False
        self.camera.hflip = False
        self.camera.framerate = 30
        self.rawCapture = PiRGBArray(self.camera)

        time.sleep(1)

    def capture_frame(self):
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            self.rawCapture.truncate(0)
            break

        return image

    def find_target(self):
		image = self.capture_frame()
		if self.record:
			original_image = image.copy()
		image = self.process_image(image)
		contours = self.find_contours(image)

		if contours is not False:
			squares, area = self.find_squares(contours)
			if self.record:
				self.draw_squares(original_image, squares)

			if squares is not None and len(squares) > 0:
				result = self.find_centre(squares[0])
			else:
				result = False

		else:
			result = False

		if self.record:
			self.out.write(original_image)

		return result

    @staticmethod
    def draw_squares(image, squares):
        for square in squares:
            cv2.drawContours(image, [square], -1, (0, 255, 0), 3)
    
    def process_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        outputs = []
        for (lower, upper) in self.boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
    
            mask = cv2.inRange(image, lower, upper)
            outputs.append(mask)
    
        image = cv2.bitwise_or(outputs[0], outputs[1])
    
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # display_images(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    
        image = cv2.threshold(image, 245, 255, cv2.THRESH_BINARY)[1]
        # display_images(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    
        return image
    
    @staticmethod
    def find_contours(image):
        conts, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if len(conts) > 0:
            return conts
        else:
            return False
    
    def find_squares(self, conts):
        if len(conts) > 0:
            squares = []
            for cont in conts:
                # calculate approximate polygons for the contours
                epsilon = 0.05 * cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, epsilon, True)
    
                # check if they're convex squares
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    squares.append(approx)
    
            # sort according to contour area and only take top 10%
            squares = sorted(squares, key=cv2.contourArea, reverse=True)
            big_squares = squares[0:int(math.ceil(len(squares) * 0.1))]
    
            # Create area threshold
            thresh = 0.03
            thresh_area = (self.height * thresh) * (self.height * thresh)
            thresh_squares = []
    
            # Check squares fit in the area threshold
            if len(big_squares) > 0:
                for square in big_squares:
                    area = cv2.contourArea(square)
                    if area >= thresh_area:
                        thresh_squares.append(square)
    
            # return area if the list of squares isn't empty
            if len(big_squares) > 0:
                area = cv2.contourArea(big_squares[0])
                return thresh_squares, area
            else:
                return thresh_squares, 0
        else:
            return None, 0
    
    @staticmethod
    def find_centre(square):
        if len(square) > 0:
            M = cv2.moments(square)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return cx, cy
