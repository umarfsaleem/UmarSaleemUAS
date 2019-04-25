from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import math

connection_string = '127.0.0.1:14551'

print(connection_string)

vehicle = connect(connection_string, wait_ready=True)

def wp_change(r_or_l):

alreadyAlerted = False
while not vehicle.armed:
    if not alreadyAlerted:
        print('waiting to arm')
        alreadyAlerted = True
    time.sleep(1)

if vehicle.armed:
    alreadyAlerted = False

while vehicle.armed:
    print 'go left(L) or go right(R)'
    direction = raw_input()
    duration = 10
    if direction == 'l' or direction == 'L':
        do_something
    elif direction == 'r' or direction == 'R':
        do_something

    else:
        print 'incorrect input'