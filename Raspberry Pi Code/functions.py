import cv2
import numpy as np
import time
import math


def mask_hsv(image):
       boundaries = [([0, 50, 60], [10, 255, 255]),
                     ([165, 50, 60], [179, 255, 255])]

       image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

       outputs = []
       for (lower, upper) in boundaries:
           lower = np.array(lower, dtype="uint8")
           upper = np.array(upper, dtype="uint8")

           mask = cv2.inRange(image, lower, upper)
           output = cv2.bitwise_and(image, image, mask=mask)
           outputs.append(output)

       output = cv2.bitwise_or(outputs[0], outputs[1])
       output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
       return output


def contour_find(image):
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       blurred = cv2.GaussianBlur(gray, (19, 19), 0)
       thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]
       conts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       return conts


def contour_draw(image, conts, color=(0, 255, 0)):
       if len(conts) > 0:
           cv2.drawContours(image, conts, -1, color, 3)

       return image
       

def quick_squares(image):
    height = image.shape[0]

    conts = quick_contours(image)

    squares, area = square_find_2(conts, height)

    return squares


def quick_contours(image):
    boundaries = [([0, 60, 60], [10, 255, 255]),
                  ([165, 60, 60], [179, 255, 255])]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    outputs = []
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        outputs.append(mask)

    image = cv2.bitwise_or(outputs[0], outputs[1])

    image = cv2.GaussianBlur(image, (3, 3), 0)
    # display_images(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    image = cv2.threshold(image, 245, 255, cv2.THRESH_BINARY)[1]
    # display_images(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    conts, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return conts


def square_find_2(conts, height):
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
        big_squares = squares[0:int(math.ceil(len(squares)*0.1))]

        # Create area threshold
        thresh = 0.018
        thresh_area = (height*thresh)*(height*thresh)
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


def calculate_centre(conts):
        if len(conts) > 0:
            M = cv2.moments(conts)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return cx, cy
