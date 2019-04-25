from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import math

connection_string = '127.0.0.1:14551'

print(connection_string)

vehicle = connect(connection_string, wait_ready=True)


def change_roll(r_or_l, d):
    if r_or_l == 'r':
        for x in range(0, d):
            vehicle.channels.overrides['1'] = 1600
            time.sleep(1)
        print('finished')
        vehicle.channels.overrides = {}
    elif r_or_l == 'l':
        for x in range(0, d):
            vehicle.channels.overrides['1'] = 1400
            time.sleep(1)
        print('finished')
        vehicle.channels.overrides = {}


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
        change_roll('l', duration)
    elif direction == 'r' or direction == 'R':
        change_roll('r', duration)

    else:
        print 'incorrect input'
