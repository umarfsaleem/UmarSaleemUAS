import numpy as np
import cv2
import time
import math


class TargetFinder:

    def __init__(self, height, width):
        print ('target finder started')
        self.boundaries = [([0, 128, 60], [10, 255, 255]),
                           ([165, 128, 60], [179, 255, 255])]

        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output', 400, 300)
        cv2.moveWindow('output', 1020, 450)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.out = cv2.VideoWriter('/Users/UmarSaleem/PycharmProjects/UAS_full_system_test/output-vid.mp4', fourcc,
                                   10.0, (int(width), int(height)))

    def find_target(self, image, display=0, record=False):
        # 0: no display, 1: display rgb with target outlines 2: display rgb and colour filtered image

        height, width = image.shape[:2]
        masked = self.__mask_hsv(image)
        contours = self.__contour_find(masked)
        squares = self.__square_find(contours, height)

        thresh = int(width / 2)
        margin = int(width * 0.1)

        if squares is not None:
            position, coordinate = self.__find_position(squares[0], thresh, margin)
        else:
            position = 'none'
            coordinate = (0, 0)

        if display != 0:
            if squares is not None:
                if len(squares) > 0:
                    for square in squares:
                        self.__contour_draw(image, [square])

            self.__side_draw(image, position, coordinate)
            cv2.line(image, (thresh, 0), (thresh, height), [255, 0, 0], 3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 2
            colour = [255, 255, 255]
            thickness = 2

            if record:
                cv2.putText(image, 'Recording', (20, 120), font, font_size, colour, thickness)

            if display == 2:
                cv2.imshow('output', np.vstack([image, masked]))
            else:
                if record:
                    self.out.write(image)
                cv2.imshow('output', image)

        return position, coordinate, height, width

    def __mask_hsv(self, image):

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        outputs = []
        for (lower, upper) in self.boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv2.inRange(hsv_image, lower, upper)
            output = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
            outputs.append(output)

        output = cv2.bitwise_or(outputs[0], outputs[1])
        output_rgb = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)

        return output_rgb

    @staticmethod
    def __contour_find(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (19, 19), 0)
        thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]
        conts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return conts

    @staticmethod
    def __square_find(conts, height):
        if len(conts) > 0:

            squares = []
            thresh = 0.00005

            for cont in conts:
                # calculate approximate polygons for the contours
                epsilon = 0.05 * cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, epsilon, True)

                # check if they're convex squares
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    squares.append(approx)

            # create threshold area to check squares against
            thresh_area = height*height*thresh

            # filter out squares which are too small
            if len(squares) > 0:

                squares = [square for square in squares if cv2.contourArea(square) > thresh_area]

            # sort squares wrt area
            squares = sorted(squares, key=cv2.contourArea, reverse=True)

            if len(squares) > 0:
                return squares

    @staticmethod
    def __contour_draw(image, conts):

        if len(conts) > 0:
            cv2.drawContours(image, conts, -1, (0, 255, 0), 3)

        return image

    def __find_position(self, square, thresh, margin):

        x, y = self.__calculate_centre(square)
        if (thresh - margin) < x < (thresh + margin):
            return 'centre', [x, y]
        elif x > thresh:
            return 'right', [x, y]
        elif x < thresh:
            return 'left', [x, y]

    @staticmethod
    def __calculate_centre(conts):

        if len(conts) > 0:
            M = cv2.moments(conts)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return cx, cy

    @staticmethod
    def __side_draw(image, position, coordinate):

        x, y = coordinate
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 2
        colour = [255, 255, 255]
        thickness = 2
        if position == 'right':
            cv2.putText(image, 'Right', (x - 20, y - 20), font, font_size, colour, thickness)
            cv2.putText(image, 'Right', (20, 60), font, font_size, colour, thickness)
        elif position == 'left':
            cv2.putText(image, 'Left', (x - 20, y - 20), font, font_size, colour, thickness)
            cv2.putText(image, 'Left', (20, 60), font, font_size, colour, thickness)
        elif position == 'centre':
            cv2.putText(image, 'Centre', (x - 20, y - 20), font, font_size, colour, thickness)
            cv2.putText(image, 'Centre', (20, 60), font, font_size, colour, thickness)
        else:
            cv2.putText(image, 'None', (20, 60), font, font_size, colour, thickness)

    def video_write_release(self):
        self.out.release()
