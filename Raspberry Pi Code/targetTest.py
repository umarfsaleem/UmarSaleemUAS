from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import functions as fn
import numpy as np
import datetime

# this function is yet untested
# first test it isn't workin

camera = PiCamera()
camera.framerate = 30
rawCapture = PiRGBArray(camera)

time.sleep(1)

def start_recorder():
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    current_time = datetime.datetime.now().strftime("%d-%m-%y-%H:%M")
    output_file = '/home/pi/scaleTestImages/video-out-' + current_time + '.avi'
    out = cv2.VideoWriter(output_file, fourcc, 7, (720, 480))
    return out

##camera.capture(rawCapture, format="bgr")
##image = rawCapture.array

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("output", 1200 , 600)
times = []
record = False

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    startTime = time.time()
    
    image = frame.array

##    Some processing of the frame
    edit_image = image.copy()
    #hsv_image = image.copy()
    #hsv_image = fn.mask_hsv(hsv_image)
    squares = fn.quick_squares(edit_image)
    
    if squares is not None:
        for square in squares: 
            output = fn.contour_draw(image, [square])
            x, y = fn.calculate_centre(square)
    
    if record: 
        cv2.putText(image, "Recording", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 3)
        out.write(image)
        
    #cv2.imshow("output", np.hstack([image, hsv_image]))
    cv2.imshow("output", image)

    rawCapture.truncate(0)

    if len(times) > 15:
        del times[0]

    times.append(time.time() - startTime)
    averageTime = sum(times) / len(times)
    fps = 1/averageTime
    #if squares is not None:
        #print "centre is at: " + str(x) + ", " + str(y) + " fps is: " + str(fps)
    #else:
    print "fps is: " + str(fps)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        current_time = datetime.datetime.now().strftime("%d-%m-%y-%H:%M")
        filename = "/home/pi/scaleTestImages/targetTest-" + current_time + ".png"
        print "image saved"
        cv2.imwrite(filename, image)
    elif cv2.waitKey(1) & 0xFF == ord('r'): 
        if record is False: 
            record = True
            print "recording started"
            out = start_recorder()
        else: 
            record = False
            print "recording finished"
            out.release()
    
cv2.destroyAllWindows()
