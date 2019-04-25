from dronekit import connect, VehicleMode, LocationGlobalRelative
import dronekit_sitl
from dronekit_sitl import SITL
from pymavlink import mavutil
import time
import math

# sitl = SITL(path='venv/bin/dronekit-sitl')
# sitl.download(sitl, 'copter', '3.3', verbose=True)
# sitl_args=['--model', 'quad', '--home=52.814589,-4.127592,0,0']
# sitl.launch(sitl_args, await_ready=True, restart=True)
# sitl = dronekit_sitl.start_default(52.814589, -4.127592)
# connection_string = sitl.connection_string()
connection_string = '127.0.0.1:14551'

print(connection_string)

vehicle = connect(connection_string, wait_ready=True)


def status_check():
    status = None
    while vehicle.system_status.state is not 'POWEROFF':

        if vehicle.system_status.state is not status or status is None:
            print(vehicle.system_status.state)
            status = vehicle.system_status.state
            time.sleep(1)


def send_body_ned_velocity(velocity_x, velocity_y, velocity_z, duration=0):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED, # frame Needs to be MAV_FRAME_BODY_NED for forward/back left/right control.
        0b0000111111000111, # type_mask
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # m/s
        0, 0, 0, # x, y, z acceleration
        0, 0)
    for x in range(0, duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)


def send_global_velocity(velocity_x, velocity_y, velocity_z, duration):
    """
    Move vehicle in direction based on specified velocity vectors.
    """
    msg = vehicle.message_factory.set_position_target_global_int_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, # lat_int - X Position in WGS84 frame in 1e7 * meters
        0, # lon_int - Y Position in WGS84 frame in 1e7 * meters
        0, # alt - Altitude in meters in AMSL altitude(not WGS84 if absolute or relative)
        # altitude above terrain if GLOBAL_TERRAIN_ALT_INT
        velocity_x, # X velocity in NED frame in m/s
        velocity_y, # Y velocity in NED frame in m/s
        velocity_z, # Z velocity in NED frame in m/s
        0, 0, 0, # afx, afy, afz acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    # send command to vehicle on 1 Hz cycle
    for x in range(0, duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)


def set_roi():

    print('roi method called')

    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()

    missionlist = []
    for cmd in cmds:
        missionlist.append(cmd)

    index = cmds.next - 1
    print('next command is: ' + str(cmds.next))
    roi = [missionlist[index].x, missionlist[index].y, missionlist[index].z]
    print('roi is: ' + str(roi))

    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_DO_SET_ROI, #command
        0, #confirmation
        0, 0, 0, 0, #params 1-4
        roi[0],
        roi[1],
        roi[2]
        )

    vehicle.send_mavlink(msg)


def set_velocity(l_or_r, velocity):
    # get current heading and velocity
    heading = vehicle.heading
    vx = vehicle.velocity[0]
    vy = vehicle.velocity[1]
    current_vel = math.sqrt((vx*vx)+(vy*vy))

    # calculate heading of right and left direction
    right_heading = heading + 90
    left_heading = heading + 270
    if right_heading > 360:
        right_heading = right_heading - 360
    if left_heading > 360:
        left_heading = left_heading - 360

    def calculate_vels(heading_direction):
        # convert heading to radians
        heading_direction_rads = math.radians(heading_direction)
        heading_rads = math.radians(heading)

        x = math.cos(heading_direction_rads)
        y = math.sin(heading_direction_rads)
        xf = math.cos(heading_rads)
        yf = math.sin(heading_rads)
        xvel = int(x*velocity) + int(xf*current_vel)
        yvel = int(y*velocity) + int(yf*current_vel)
        return xvel, yvel

    if l_or_r == 'right':
        return calculate_vels(right_heading)
    elif l_or_r == 'left':
        return calculate_vels(left_heading)


alreadyAlerted = False
while not vehicle.armed:
    if not alreadyAlerted:
        print('waiting to arm...')
        alreadyAlerted = True
    time.sleep(1)

if vehicle.armed:
    alreadyAlerted = False

while vehicle.armed:
    print 'go left(L) or go right(R)'
    direction = raw_input()
    duration = 10
    if direction == 'l' or direction == 'L':
        velocity_x, velocity_y = set_velocity('left', 10)
        velocity_z = 0
        vehicle.mode = VehicleMode("GUIDED")
        set_roi()
        send_global_velocity(velocity_x, velocity_y, velocity_z, duration)
        vehicle.mode = VehicleMode("AUTO")

    elif direction == 'r' or direction == 'R':
        velocity_x, velocity_y = set_velocity('right', 10)
        velocity_z = 0
        vehicle.mode = VehicleMode("GUIDED")
        set_roi()
        send_global_velocity(velocity_x, velocity_y, velocity_z, duration)
        set_roi()
        vehicle.mode = VehicleMode("AUTO")
        set_roi()

    else:
        print 'incorrect input'


