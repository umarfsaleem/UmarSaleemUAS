from imutils.video import VideoStream
import cv2
import time

vs = VideoStream(usePiCamera=True).start()
time.sleep(2)
print "video stream started"

startTime = time.time()

while (time.time() - startTime < 10):
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stream", 600, 600)
    cv2.imshow("Stream", gray)

cv2.destroyAllWindows()
vs.stop()

    
