from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import datetime

camera = PiCamera()
camera.vflip = True
camera.framerate = 30
rawCapture = PiRGBArray(camera)

time.sleep(1)

fourcc = cv2.VideoWriter_fourcc(*'H264')
current_time = datetime.datetime.now().strftime("%d-%m-%y-%H:%M")
output_file = '/home/pi/UAS/FlightVideos/' + current_time + '.avi'
out = cv2.VideoWriter(output_file, fourcc, 10, (720, 480))


##camera.capture(rawCapture, format="bgr")
##image = rawCapture.array

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("output", 600 , 600)
times = []

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    startTime = time.time()
    
    image = frame.array
    out.write(image)
    cv2.imshow("output", image)

    rawCapture.truncate(0)

    if len(times) > 15:
        del times[0]

    times.append(time.time() - startTime)
    averageTime = sum(times) / len(times)
    fps = 1/averageTime
    print "fps is: " + str(fps)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cv2.destroyAllWindows()
