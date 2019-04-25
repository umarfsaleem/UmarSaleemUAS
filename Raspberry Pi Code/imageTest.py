from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import argparse

parser = argparse.ArgumentParser(description='Take image and save it to a file')
parser.add_argument('-f', help = "please add a filename")
parser.add_argument('-w', help = "specify a time to wait before taking the image")

args = parser.parse_args()
if args.f is not None: 
	filename = args.f
	
if args.w is not None: 
	wait_time = int(args.w)
	time.sleep(wait_time)
	print "image in: "
	while wait_time > 0:
		print wait_time
		time.sleep(1)
		wait_time -= 1

camera = PiCamera()
rawCapture = PiRGBArray(camera)

time.sleep(1)

camera.capture(rawCapture, format="bgr")
image = rawCapture.array

if filename is not None:
	write_filename = filename + ".png"
	cv2.imwrite(write_filename, image)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("output", 600 , 600)
cv2.imshow("output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
