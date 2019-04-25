import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename as open_file
from imutils.video import FileVideoStream
from queue import Queue
from threading import Thread
import time
import math
import pytesseract
from imutils.object_detection import non_max_suppression
from PIL import Image


def character_detect(image):
    og = image.copy()
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 600, 600)
    cv2.imshow("output", og)
    cv2.waitKey(0)
    squares = quick_squares(image)
    rect = cv2.minAreaRect(squares[0])
    crop = crop_minAreaRect(og, rect)
    cv2.imshow("output", crop)
    cv2.waitKey(0)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.threshold(crop, 150, 255, cv2.THRESH_BINARY_INV)[1]
    # crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    cv2.imshow("output", crop)
    cv2.waitKey(0)
    config = "-l eng --oem 1 --psm 10"
    text = pytesseract.image_to_string(crop, config=config)
    image_text = "Letter is: " + text
    cv2.putText(og, image_text, (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, [0, 255, 255], 30)
    contour_draw(og, [squares[0]])
    print(image_text)
    cv2.imshow("output", og)

    if cv2.waitKey(0) & 0xFF == ord('s'):
        filename = "letter-detect-" + text + ".JPG"
        cv2.imwrite(filename, og)
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()


def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop


def steps(img):
    # make copy of original image for display at the end
    og_img = img.copy()

    # apply hsv mask and return hsv image as well as masked image converted back to bgr
    boundaries = [([0, 128, 60], [10, 255, 255]),
                  ([165, 128, 60], [179, 255, 255])]
    for_hsv = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    outputs = []
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        filtered = cv2.inRange(hsv, lower, upper)
        outputs.append(filtered)

    filtered = cv2.bitwise_or(outputs[0], outputs[1])
    ffiltered = filtered.copy()
    ffiltered = cv2.cvtColor(ffiltered, cv2.COLOR_GRAY2BGR)

    hsv, hsv2bgr = mask_hsv(for_hsv)

    # blur grayscale image
    blurred = cv2.GaussianBlur(filtered, (19, 19), 0)
    fblurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    # threshold blurred grayscale image
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    fthresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # create copy of masked image to perform contour detection on (does these previous steps again)
    ct_preimg = fthresh.copy()
    contours = contour_find(ct_preimg)

    # draw contours on masked image
    ct_img = fthresh.copy()
    contour_draw(ct_img, contours)

    # create copy of masked image and find squares then draw on image
    sq_img = fthresh.copy()
    height = sq_img.shape[0]
    squares, area = square_find_2(contours, height)
    if squares is not None:
        for square in squares:
            contour_draw(sq_img, [square])

    # create copy of original image and draw squares on that
    final_img = og_img.copy()
    if squares is not None:
        for square in squares:
            contour_draw(final_img, [square])
            x, y = calculate_centre(square)
            cv2.circle(final_img, (x, y), 40, [0, 255, 255], -1)
            y = y - 100
            cv2.putText(final_img, "Centre", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5, [0, 255, 255], 30)

    # display all images
    images = [og_img, hsv2bgr, ffiltered, fblurred, fthresh, ct_img, sq_img, final_img]
    output_img = np.hstack(images)
    cv2.imwrite('/Users/UmarSaleem/PycharmProjects/ShapeID/image-process-steps.png', output_img)
    display_images(images)


def test(image):
    gamma = 1
    # gamma = float(input("put gamma value"))
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    g_image = cv2.LUT(image, table)
    squares = quick_squares(g_image)
    if squares is not None:
        for square in squares:
            contour_draw(g_image, [square])
    display_images([g_image])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    display_images([image])


def surrounding_circle(img, square):

    blank = np.zeros((img.shape[0], img.shape[1]), img.dtype)

    circle = cv2.minEnclosingCircle(square)
    cv2.circle(blank, (int(circle[0][0]), int(circle[0][1])), int(circle[1]), 255, -1)
    cv2.drawContours(blank, [square], -1, 0, -1)

    output = cv2.bitwise_and(img, img, mask = blank)
    cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    pvalues = output[np.nonzero(output)]
    avg = pvalues.mean()

    thresh = 150

    if avg > thresh:
        return True
    else:
        return False


def quick_squares(image):
    height = image.shape[0]

    conts = quick_contours(image)

    squares, area = square_find_2(conts, height)

    return squares


def quick_contours(image):
    boundaries = [([0, 128, 60], [10, 255, 255]),
                  ([165, 128, 60], [179, 255, 255])]

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


def mask(image):
        boundaries = [([2, 2, 100], [50, 56, 255])]
        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)
        return output


def mask_hsv(image):
        boundaries = [([0, 128, 60], [10, 255, 255]),
                      ([165, 128, 60], [179, 255, 255])]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        outputs = []
        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv2.inRange(image, lower, upper)
            output = cv2.bitwise_and(image, image, mask=mask)
            outputs.append(output)

        output = cv2.bitwise_or(outputs[0], outputs[1])
        output_rgb = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
        return output, output_rgb


def contour_find(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (19, 19), 0)
        thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]
        # display_images([gray, thresh])
        conts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return conts


def poly_approx(conts):
        if len(conts) > 0:
            conts = sorted(conts, key = cv2.contourArea, reverse = True)
            epsilon = 0.1 * cv2.arcLength(conts[0], True)
            approx = cv2.approxPolyDP(conts[0], epsilon, True)
            return approx
        else:
            return np.empty(0)


def square_find(conts):
    if len(conts) > 0:
        conts = sorted(conts, key=cv2.contourArea, reverse=True)
        new_conts = conts[0:math.ceil(len(conts)*0.05)]
        squares = []
        for cont in new_conts:
            epsilon = 0.1*cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, epsilon, True)
            if len(approx) > 3:
                squares.append(approx)

        return squares


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
        big_squares = squares[0:math.ceil(len(squares)*0.1)]

        # Create area threshold
        thresh = 0.03
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


def square_find_3(conts, height, frame):
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
        big_squares = squares[0:math.ceil(len(squares)*0.1)]

        # Create area threshold
        thresh_area = (height*0.05)*(height*0.05)
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


def contour_draw(image, conts, color=(0, 255, 0)):
        if len(conts) > 0:
            cv2.drawContours(image, conts, -1, color, 3)

        return image


def display_images(images):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 600, 600)
        cv2.imshow("output", np.hstack(images))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def plot_histogram_hsv(image, title='Histogram', mask=None):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        chans = cv2.split(hsv_image)
        colours = ("b", "g", "r")
        plt.figure()
        plt.title(title)
        plt.xlabel('Bins')
        plt.ylabel('no. of Pixels')

        for (chan, colour) in zip(chans, colours):
            hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
            plt.plot(hist, color=colour)
            plt.xlim([0, 256])

        plt.legend(['Hue', 'Saturation', 'Value'])
        plt.show()


def select_file():
        print('Write file name:')
        filename = input()
        full_filename = "/Users/UmarSaleem/PycharmProjects/ShapeID/" + filename
        image = cv2.imread(full_filename)
        return image


def select_file_gui():
        file = open_file()
        image = cv2.imread(file)
        return image


def calculate_centre(conts):
        if len(conts) > 0:
            M = cv2.moments(conts)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return cx, cy


def select_video_file_gui():
    file = open_file()
    return file


def contours(image):
    output = mask(image)
    contours = contour_find(output)
    output = contour_draw(output, contours)
    images = [image, output]
    display_images(images)


def hsv_contours(image):
    start = time.time()
    output, output_rgb = mask_hsv(image)
    contours = contour_find(output_rgb)
    elapsed = time.time() - start
    print("time taken for hsv_contours: " + str(elapsed))
    output_rgb = contour_draw(output_rgb, contours)
    images = [image, output_rgb]
    display_images(images)


def rgb_mask(image):
    output = mask(image)
    images = [image, output]
    display_images(images)


def hsv_mask(image):
    output, output_rgb = mask_hsv(image)
    images = [image, output, output_rgb]
    display_images(images)


def find_shapes(image):
    output = mask(image)
    contours = contour_find(output)
    poly = poly_approx(contours)
    contour_draw(output, [poly])
    images = [image, output]
    display_images(images)


def hsv_find_shapes(image):
    output, output_rgb = mask_hsv(image)
    contours = contour_find(output_rgb)
    height = image.shape[0]
    squares, area = square_find_2(contours, height)
    for square in squares:
        contour_draw(output_rgb, [square])
    images = [image, output_rgb]
    display_images(images)


def play_video(input_file):

    stream = cv2.VideoCapture(input_file)
    stopped = False

    Q = Queue(maxsize=64)

    def update():

        while True:

            if stopped:
                return

            if not Q.full():
                grabbed, load_frame = stream.read()

                if not grabbed:
                    return

                Q.put(load_frame)

    t = Thread(target=update)
    t.daemon = True
    t.start()
    time.sleep(1)

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 600, 600)

    while Q.qsize() > 0:

        frame = Q.get()
        cv2.imshow('output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()
    stopped = True


def play_video_imutils(input_file):
    fvs = FileVideoStream(input_file).start()
    time.sleep(1)

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 600, 600)

    fps = fvs.stream.get(cv2.CAP_PROP_FPS)
    tpf = int(1000/fps)

    while fvs.more():
        frame = fvs.read()
        cv2.imshow('output', frame)
        if cv2.waitKey(tpf) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    fvs.stop()


def filter_video(input_file):
    fvs = FileVideoStream(input_file).start()
    time.sleep(1)

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 600, 600)

    fps = fvs.stream.get(cv2.CAP_PROP_FPS)
    tpf = int(1000 / fps)

    while fvs.more():
        frame = fvs.read()
        output, filtered = mask_hsv(frame)
        cv2.imshow('output', filtered)
        if cv2.waitKey(tpf) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    fvs.stop()


def contour_video(input_file):
    fvs = FileVideoStream(input_file).start()
    time.sleep(1)

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 600, 600)

    fps = fvs.stream.get(cv2.CAP_PROP_FPS)
    tpf = int(1000 / fps)

    while fvs.more():
        frame = fvs.read()
        output, filtered = mask_hsv(frame)
        contours = contour_find(filtered)
        filtered = contour_draw(filtered, contours)
        cv2.imshow('output', filtered)
        if cv2.waitKey(tpf) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    fvs.stop()


def shape_find_video(input_file):
    fvs = FileVideoStream(input_file).start()
    time.sleep(1)

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 600, 600)

    fps = fvs.stream.get(cv2.CAP_PROP_FPS)
    tpf = int(1000 / fps)

    width = fvs.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = fvs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('/Users/UmarSaleem/PycharmProjects/ShapeID/output-vid.mp4', fourcc, fps, (int(width*2), int(height)))

    fps_average = []

    while fvs.more():
        frame = fvs.read()
        original = frame.copy()
        start_time = time.time()
        height, width = frame.shape[:2]
        output, filtered = mask_hsv(frame)
        contours = contour_find(filtered)

#        poly = poly_approx(contours)

#        height, width = frame.shape[:2]
#        thresh = int(width / 2)
#        cv2.line(frame, (thresh, 0), (thresh, height), [255, 0, 0], 3)

#        if len(poly) == 4:
#            contour_draw(frame, [poly])
#            x, y = calculate_centre(poly)
#            cv2.circle(frame, (x, y), 7, [0, 255, 255], -1)
#            if x > thresh:
#                cv2.putText(frame, 'Right', (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
#            elif x < thresh:
#                cv2.putText(frame, 'Left', (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)

        squares, area = square_find_2(contours, height)
        # cv2.putText(frame, str(area), (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2, bottomLeftOrigin=True)

        thresh = int(width / 2)
        cv2.line(frame, (thresh, 0), (thresh, height), [255, 0, 0], 1)

        if squares is not None:
            for square in squares:
                # wsurround = surrounding_circle(frame, square)
                # if wsurround:
                #     contour_draw(frame, [square])
                # else:
                #     contour_draw(frame, [square], color = (255, 0, 0))
                contour_draw(frame, [square])
                x, y = calculate_centre(square)
                cv2.circle(frame, (x, y), 7, [0, 255, 255], -1)
                if x > thresh:
                    cv2.putText(frame, 'Right', (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
                elif x < thresh:
                    cv2.putText(frame, 'Left', (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)

        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 1/elapsed_time
        fps_text = "FPS: " + str(fps)
        # cv2.putText(frame, fps_text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
        fps_average.append(fps)

        out.write(np.hstack([original, frame]))
        cv2.imshow('output', np.hstack([original, frame]))
        if cv2.waitKey(tpf) & 0xFF == ord('q'):
            break

    average_fps = sum(fps_average)/len(fps_average)
    print("FPS: " + str(average_fps))
    cv2.destroyAllWindows()
    fvs.stop()


def video_test(input_file):
    fvs = FileVideoStream(input_file).start()
    time.sleep(1)
    # tracker = cv2.TrackerMOSSE_Create()

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 600, 600)

    fps = fvs.stream.get(cv2.CAP_PROP_FPS)
    tpf = int(1000 / fps)

    fps_average = []

    while fvs.more():
        frame = fvs.read()
        original = frame.copy()
        start_time = time.time()
        height, width = frame.shape[:2]

        # contours = quick_contours(frame)
        squares = quick_squares(frame)

        thresh = int(width / 2)
        cv2.line(frame, (thresh, 0), (thresh, height), [255, 0, 0], 1)

        # contour_draw(frame, contours)

        if squares is not None:
            for square in squares:
                contour_draw(frame, [square])
                x, y = calculate_centre(square)
                cv2.circle(frame, (x, y), 7, [0, 255, 255], -1)
                if x > thresh:
                    cv2.putText(frame, 'Right', (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
                elif x < thresh:
                    cv2.putText(frame, 'Left', (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)

        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 1 / elapsed_time
        fps_text = "FPS: " + str(fps)
        # cv2.putText(frame, fps_text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
        fps_average.append(fps)

        cv2.imshow('output', np.hstack([original, frame]))
        if cv2.waitKey(tpf) & 0xFF == ord('q'):
            break

    average_fps = sum(fps_average) / len(fps_average)
    print("FPS: " + str(average_fps))
    cv2.destroyAllWindows()
    fvs.stop()


def webcam():
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 1200, 600)

    while cap.isOpened():
        grabbed, frame = cap.read()

        if grabbed:
            height, width = frame.shape[:2]
            output, filtered = mask_hsv(frame)
            contours = contour_find(filtered)
            squares, area = square_find_2(contours, height)
            cv2.putText(frame, str(area),  (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2, bottomLeftOrigin=True)

            contour_draw(filtered, contours)

            thresh = int(width / 2)
            cv2.line(frame, (thresh, 0), (thresh, height), [255, 0, 0], 3)

            for square in squares:
                contour_draw(frame, [square])
                x, y = calculate_centre(square)
                cv2.circle(frame, (x, y), 7, [0, 255, 255], -1)
                if x > thresh:
                    cv2.putText(frame, 'Right', (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
                elif x < thresh:
                    cv2.putText(frame, 'Left', (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)

            cv2.imshow('output', np.hstack([frame, filtered]))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

