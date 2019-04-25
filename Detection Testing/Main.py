import Operations as op
import tkinter as tk
import cv2

input_image = '/Users/UmarSaleem/PycharmProjects/ShapeID/stock-image.jpg'
video_file = '/Users/UmarSaleem/PycharmProjects/ShapeID/stock-video.m4v'


def pick_file():
    global input_image
    input_image = op.select_file_gui()


def pick_video_file():
    global video_file
    video_file = op.select_video_file_gui()


def test():
    op.test(input_image)


def rgb_mask():
    op.rgb_mask(input_image)


def hsv_mask():
    op.hsv_mask(input_image)


def contours():
    op.contours(input_image)


def hsv_contours():
    op.hsv_contours(input_image)


def find_shapes():
    op.find_shapes(input_image)


def hsv_find_shapes():
    op.hsv_find_shapes(input_image)


def steps():
    op.steps(input_image)


def play_video():
    op.play_video(video_file)


def play_video_imutils():
    op.play_video_imutils(video_file)


def filter_video():
    op.filter_video(video_file)


def contour_video():
    op.contour_video(video_file)


def shape_find_video():
    op.shape_find_video(video_file)


def video_test():
    op.video_test(video_file)


def webcam():
    op.webcam()


def character_detect():
    op.character_detect(input_image)


def setup():
    global input_image
    input_image = cv2.imread(input_image)
    master = tk.Tk()

    buttons = []

    title = tk.Label(text = 'Which program would you like to run?')
    pick_file_button = tk.Button(text = 'Pick a File', command = pick_file)
    buttons.append(pick_file_button)
    pick_video_file_button = tk.Button(text = 'Pick a Video File', command = pick_video_file)
    buttons.append(pick_video_file_button)
    test_button = tk.Button(text = 'Test', command = test)
    buttons.append(test_button)
    rgb_mask_button = tk.Button(text = 'RGB Mask', command = rgb_mask)
    buttons.append(rgb_mask_button)
    hsv_mask_button = tk.Button(text = 'HSV Mask', command = hsv_mask)
    buttons.append(hsv_mask_button)
    contour_button = tk.Button(text = 'Find Contours', command = contours)
    buttons.append(contour_button)
    hsv_contour_button = tk.Button(text = 'Find HSV Contours', command = hsv_contours)
    buttons.append(hsv_contour_button)
    shape_find_button = tk.Button(text = 'Find Shapes', command = find_shapes)
    buttons.append(shape_find_button)
    hsv_shape_find_button = tk.Button(text = 'Find HSV Shapes', command = hsv_find_shapes)
    buttons.append(hsv_shape_find_button)
    steps_button = tk.Button(text = 'Full process showing steps', command = steps)
    buttons.append(steps_button)
    play_video_button = tk.Button(text = 'Play Video', command = play_video)
    buttons.append(play_video_button)
    play_video_imutils_button = tk.Button(text = 'Play Video (imutils)', command = play_video_imutils)
    buttons.append(play_video_imutils_button)
    filter_video_button = tk.Button(text = 'HSV Filter Video', command = filter_video)
    buttons.append(filter_video_button)
    contour_video_button = tk.Button(text = 'HSV Contour Video', command = contour_video)
    buttons.append(contour_video_button)
    shape_find_video_button = tk.Button(text = 'HSV Shape Find Video', command = shape_find_video)
    buttons.append(shape_find_video_button)
    video_test_button = tk.Button(text = 'Video Test', command = video_test)
    buttons.append(video_test_button)
    webcam_button = tk.Button(text = 'Webcam', command = webcam)
    buttons.append(webcam_button)
    text_detect_button = tk.Button(text = 'Text Detection', command = character_detect)
    buttons.append(text_detect_button)

    title.grid(row=0)

    i = 1
    for button in buttons:
        button.grid(row=i)
        i += 1

    master.title("UAV Object Detection")
    master.mainloop()


setup()

# inputFalse = True

# while inputFalse:
#     print("which program would you like to run?")
#     print("RGB Mask (M)")
#     print("HSV Mask (N)")
#     print("Find Contours (C)")
#     print("Approximate Polygon (P)")
#     print("Create Histogram (H)")
#     print("")
#     option = input()
#     if option == "M":
#         output = op.mask(input_image)
#         images = [input_image, output]
#         op.display_images(images)
#         inputFalse = False
#     else:
#         print("false input")







