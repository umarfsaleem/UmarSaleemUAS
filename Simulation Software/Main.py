import time
import cv2
import screen_capture
import target_find
import vehicle

sc = screen_capture.ScreenCapture()
sc.start()

finder = target_find.TargetFinder(sc.height, sc.width)

ap = vehicle.Autopilot()

record = False
times = []
while True:
    start_time = time.time()

    image = sc.capture(display=False)
    position, coordinate, sc_height, sc_width = finder.find_target(image, display=1, record=record)
    ap.direct(direction=position, coordinate=coordinate, sc_height=sc_height, sc_width=sc_width)

    end_time = time.time()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)

    if cv2.waitKey(25) & 0xFF == ord('p'):
        paused = True
        while paused:
            time.sleep(1)
            if cv2.waitKey(25) & 0xFF == ord('p'):
                paused = False

    if cv2.waitKey(25) & 0xFF == ord('r'):
        if record:
            finder.video_write_release()
            record = False
        else:
            record = True

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        average_time = sum(times)/len(times)
        fps = 1/average_time
        print('average fps was: ' + str(fps))
        break
