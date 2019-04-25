from picamera import PiCamera
import time

camera = PiCamera()

camera.start_recording('exampleVid.h264')
time.sleep(10)
camera.stop_recording()
