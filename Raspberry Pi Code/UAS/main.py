import targetFind
import vehicle
import time
import argparse
import thread
import datetime

current_time = datetime.datetime.now().strftime("%d-%m-%y-%H:%M")
log_file = open("/home/pi/UAS/FlightLogs/" + current_time + ".txt", "w+")

parser = argparse.ArgumentParser(description='Provide arguments')
parser.add_argument('-s', action='store_true', default=False, 
		help='pass -s if you want to save the video feed')
parser.add_argument('-g', action='store_false', default=True, 
		help='pass -g to turn guidance off and only do video part')

log_file.write("Program started at " + current_time + "\n\n")

keep_going = True
def get_input():
	global keep_going
	response = raw_input()
	if response == 'q': 
		keep_going = False
	else: 
		get_input()

args = parser.parse_args()
record = args.s
guide = args.g

if record: 
	log_file.write("recording video: on \n")
else: 
	log_file.write("recording video: off \n")

finder = targetFind.TargetFinder(record)
if guide: 
	autopilot = vehicle.Autopilot(finder.height, finder.width)
	log_file.write("autopilot guidance: on \n\n")
else: 
	log_file.write("autopilot guidance: off \n\n")
	
thread.start_new_thread(get_input, ())

count = 0
start_time = time.time()
while keep_going: 
	frame_time = time.time()
	
	count += 1
	log_file.write("Frame " + str(count) + ": \n")

	centre = finder.find_target()
	
	result = False
	if centre is not False: 
		print "target found at: " + str(centre)
		log_file.write("\t target found at: " + str(centre) + "\n")
		
		if guide: 
			result = autopilot.direct(centre)
	else:
		print "no target found"
		log_file.write("\t no target found \n") 
		
	if result is False : 
		print "no coordinates sent"
		log_file.write("\t no coordinates sent \n")
	else: 
		log_file.write("\t drone sent to: " + str(result) + "\n")
	
	fps = 1/(time.time() - frame_time)
	print "FPS: " + str(fps)
	log_file.write("\t FPS: " + str(fps) + "\n")

if record: 
	finder.out.release()
