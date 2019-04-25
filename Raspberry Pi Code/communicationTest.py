from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil 
import time
import math

connection_string = '/dev/ttyS0'

vehicle = connect(connection_string, wait_ready=True, baud=57600)

start_time = time.time()

while (time.time() - start_time) < 60:
	status = vehicle.system_status.state
	print "vehicle state is: " + status
	mode = vehicle.mode.name
	print "vehicle mode is: " + mode
	time.sleep(2)
	
	count = 1
	if ((time.time() - start_time)/10) > count: 
		count = count + 1
		if vehicle.mode.name is "GUIDED":
			vehicle.mode = VehicleMode("STABILIZE")
		else: 
			vehicle.mode = VehicleMode("GUIDED")
			
vehicle.close()
