from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import math


class Autopilot:

    def __init__(self):
        print ('autopilot communication started')
        self.connection_string = '127.0.0.1:14551'
        self.v = connect(self.connection_string, wait_ready=True)
        print ('vehicle connect finished')
        self.wait_to_arm()
        self.none_count = 10
        self.t_gps_coords = []
        self.target_gps_coords = None
        self.centre_override_used = False
        self.target_found = False
        self.none_start = 0.0
        self.t_bearings = []

    def direct(self, direction, coordinate=None, sc_height=None, sc_width=None):

        vehicle = self.v

        if direction == 'none':
            self.none_count += 1

            # if there has been no target found for 3 seconds revert to auto mode
            if self.target_found:
                self.none_start = time.time()
                self.target_found = False
            none_time = time.time() - self.none_start
            if none_time > 15.0:
                vehicle.mode = VehicleMode("AUTO")

        # Remove False to make this function get called again
        if direction != 'none' and False:
            vx, vy = self.set_velocity(direction, 10)
            vz = 0
            vehicle.mode = VehicleMode("GUIDED")
            # self.set_roi()
            self.send_global_velocity(vx, vy, vz)
            # vehicle.mode = VehicleMode("AUTO")
        elif self.none_count > 10 and False:
            vehicle.mode = VehicleMode("AUTO")

        if direction != 'none':
            self.target_found = True
            # self.target_gps_coords = self.gps_find(direction, coordinate, sc_height, sc_width, bearing=True)
            self.target_gps_coords = self.bearing_find(coordinate, sc_height, sc_width)

            if self.target_gps_coords is not None:
                print('target_gps_coords are: ' + str(self.target_gps_coords))
                mode_name = vehicle.mode.name
                if mode_name != 'GUIDED':
                    vehicle.mode = VehicleMode("GUIDED")
                    print 'mode changed to guided'
                else:
                    print 'mode already guided'
                    
                lat = self.target_gps_coords[0]
                lon = self.target_gps_coords[1]
                vehicle.simple_goto(LocationGlobalRelative(lat, lon, vehicle.location.global_frame.alt))

    def set_velocity(self, l_or_r, velocity):
        # get current heading and velocity
        heading = self.v.heading
        vx = self.v.velocity[0]
        vy = self.v.velocity[1]
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

    def set_roi(self):

        cmds = self.v.commands
        cmds.download()
        cmds.wait_ready()

        missionlist = []
        for cmd in cmds:
            missionlist.append(cmd)

        index = cmds.next - 1
        roi = [missionlist[index].x, missionlist[index].y, missionlist[index].z]

        msg = self.v.message_factory.command_long_encode(
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_CMD_DO_SET_ROI,  # command
            0,  # confirmation
            0, 0, 0, 0,  # params 1-4
            roi[0],
            roi[1],
            roi[2]
            )

        self.v.send_mavlink(msg)

    def send_global_velocity(self, velocity_x, velocity_y, velocity_z):
        """
        Move vehicle in direction based on specified velocity vectors.
        """
        msg = self.v.message_factory.set_position_target_global_int_encode(
            0,  # time_boot_ms (not used)
            0, 0,  # target system, target component
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,  # frame
            0b0000111111000111,  # type_mask (only speeds enabled)
            0,  # lat_int - X Position in WGS84 frame in 1e7 * meters
            0,  # lon_int - Y Position in WGS84 frame in 1e7 * meters
            0,  # alt - Altitude in meters in AMSL altitude(not WGS84 if absolute or relative)
            # altitude above terrain if GLOBAL_TERRAIN_ALT_INT
            velocity_x,  # X velocity in NED frame in m/s
            velocity_y,  # Y velocity in NED frame in m/s
            velocity_z,  # Z velocity in NED frame in m/s
            0, 0, 0,  # afx, afy, afz acceleration (not supported yet, ignored in GCS_Mavlink)
            0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

        self.v.send_mavlink(msg)

    def wait_to_arm(self):

        already_alerted = False
        while not self.v.armed:
            if not already_alerted:
                print('waiting to arm...')
                already_alerted = True
            time.sleep(1)

        print('Armed')

    def gps_find(self, position, t_coord, sc_height, sc_width, debug=False, bearing=False):

        # return if values haven't been assigned
        if t_coord is None or sc_height is None or sc_width is None:
            return

        # check drone is level enough as it messes up calculations if not
        roll = self.v.attitude.roll
        thresh = 0.0873  # 5 degrees in radians
        if not thresh > roll > -thresh:
            return

        # get some values from the drone that are needed
        pitch = self.v.attitude.pitch * -1
        altitude = self.v.location.global_frame.alt
        heading = math.radians(self.v.heading)
        lat = self.v.location.global_frame.lat
        lon = self.v.location.global_frame.lon

        # define conversion ratios for degrees to metres
        lat_rad = math.radians(lat)
        deg_2_m_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad) - 0.0023 * math.cos(6 * lat_rad)
        deg_2_m_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad) + 0.118 * math.cos(5 * lat_rad)

        # remove False to make this function run again, currently disabled to test bearing guidance
        if position is 'centre' and False:
            length_x = 0.0
            length_y = 100.0
        else:
            # assign FOV angles and convert to radians
            v_fov = math.radians(21.6)  # 21.6 degrees in radians
            h_fov = math.radians(30.6)  # 30.6 degrees in radians

            # alpha is angle between upper FOV line and target
            alpha_y = (float(t_coord[1]) / float(sc_height)) * v_fov

            # beta is angle between vertical and line from plane to horizontal line target is on
            beta_y = math.pi/2 - alpha_y - pitch + v_fov/2

            # change beta y to 90 degrees if it's above so there's no negative length y values
            if beta_y > math.pi/2:
                beta_y = math.pi/2

            # length y is length on the ground from drone to target straight ahead
            length_y = altitude * math.tan(beta_y)
            # length h is distance directly from plane to target
            length_h = altitude / math.cos(beta_y)

            if length_y < 0:
                length_y = length_y * -1

            if length_h < 0:
                length_h = length_h * -1

            # alpha x is angle between direct line to horizontal and line direct to target
            alpha_x = ((float(t_coord[0]) / float(sc_width)) - 0.5) * h_fov

            # negative length_x is left, positive is right
            length_x = length_h * math.tan(alpha_x)

        # direct length on the ground from plane to target
        length = math.sqrt((length_y * length_y) + (length_x * length_x))

        # bearing from north line to line on the ground from drone to target
        t_bearing = math.atan(length_x / length_y) + heading

        # override if just using bearing values not exact gps values
        if bearing is True:
            length = 200.0

        # functions to calculate difference in degrees of lat and lon from plane to target
        if t_bearing < math.pi/2:
            # north east quadrant
            length_n = length * math.cos(t_bearing)
            length_e = length * math.sin(t_bearing)
            delta_lat = length_n / deg_2_m_lat
            delta_lon = length_e / deg_2_m_lon

        elif math.pi/2 <= t_bearing < math.pi:
            # south east quadrant
            length_n = length * math.cos(math.pi - t_bearing)
            length_e = length * math.sin(math.pi - t_bearing)
            delta_lat = (length_n / deg_2_m_lat) * -1
            delta_lon = length_e / deg_2_m_lon

        elif math.pi <= t_bearing < (3 * math.pi/2):
            # south west quadrant
            length_n = length * math.cos(t_bearing - math.pi)
            length_e = length * math.sin(t_bearing - math.pi)
            delta_lat = (length_n / deg_2_m_lat) * -1
            delta_lon = (length_e / deg_2_m_lon) * -1

        else:
            # north west quadrant
            length_n = length * math.cos((2 * math.pi) - t_bearing)
            length_e = length * math.sin((2 * math.pi) - t_bearing)
            delta_lat = length_n / deg_2_m_lat
            delta_lon = (length_e / deg_2_m_lon) * -1

        # calculate gps coords of target using differences and plane gps position
        t_lat = lat + delta_lat
        t_lon = lon + delta_lon
        t_gps_coord = (t_lat, t_lon)

        # override to use just bearing values
        if bearing is True:

            if len(self.t_bearings) > 10:
                del self.t_bearings[0]
            self.t_bearings.append(t_bearing)

            print 'target bearing is' + str(t_bearing)

            # degrees in radians
            bearing_thresh = math.radians(0.5)

            upper = t_bearing + bearing_thresh
            lower = t_bearing - bearing_thresh

            thresh_count = 0
            for bearing in self.t_bearings:
                if lower < bearing < upper:
                    thresh_count += 1

            if thresh_count >= 5:
                return t_gps_coord

            # if bearing readings are within 1 degrees of each other send gps coords to be directed toward
            # remove false to make function work again
            if math.fabs(max(self.t_bearings) - min(self.t_bearings)) <= bearing_thresh \
                    and len(self.t_bearings) > 5 and False:
                return t_gps_coord

        # add target gps to list of values
        if len(self.t_gps_coords) > 5:
            del self.t_gps_coords[0]
        self.t_gps_coords.append(t_gps_coord)

        # extract lats and longs from target gps coordinates
        lats = []
        lons = []
        for coord in self.t_gps_coords:
            lats.append(coord[0])
            lons.append(coord[1])

        if debug:
            print('target coordinate in image are: ' + str(t_coord))
            print('screen height is: ' + str(sc_height))
            print('screen width is: ' + str(sc_width))
            print('altitude is: ' + str(altitude) + ' metres')
            print('pitch is: ' + str(pitch))

            if position is not 'centre':
                print('alpha y is: ' + str(alpha_y))
                print('beta y is: ' + str(beta_y))
                print('v_fov is: ' + str(v_fov))

                print('h_fov is: ' + str(h_fov))
                print('length_h is: ' + str(length_h))
                print('alpha_x is: ' + str(alpha_x))

            print('target is ' + str(length_y) + ' metres ahead.')
            print('target is ' + str(length_x) + ' metres to the left or right.')

        # only return current target coordinate if last 5 are within a reasonable threshold
        gps_thresh = 0.0001
        if math.fabs(max(lats) - min(lats)) <= gps_thresh\
                and math.fabs(max(lons) - min(lons)) <= gps_thresh\
                and len(self.t_gps_coords) > 5 \
                or position is 'centre' and not self.centre_override_used and len(self.t_gps_coords) > 5:
            if position is 'centre':
                print 'centre position override used'
                self.centre_override_used = True
            return t_gps_coord

    def bearing_find(self, ti_coord, sc_height, sc_width):

        # return if values haven't been assigned
        if ti_coord is None or sc_height is None or sc_width is None:
            return

        length = 300.0

        # check drone is level enough as it messes up calculations if not
        roll = self.v.attitude.roll
        thresh = math.radians(5.0)  # 5 degrees in radians
        if not thresh > roll > -thresh:
            return

        # get some values from the drone that are needed
        # to calculate bearing from current heading
        pitch = self.v.attitude.pitch * -1  # invert so positive pitch is pointed downward

        # get some values from the drone that are needed
        # to calculate gps coordinates to fly toward
        heading = math.radians(self.v.heading)
        lat = self.v.location.global_frame.lat
        lon = self.v.location.global_frame.lon

        # assign FOV angles and convert to radians
        v_fov = math.radians(21.6)  # 21.6 degrees in radians
        h_fov = math.radians(30.6)  # 30.6 degrees in radians

        # extract x and y coordinates of target in image
        x = float(ti_coord[0])
        y = float(ti_coord[1])

        # calculate relative angle of target from current aircraft position
        top = math.tan(((x / float(sc_width)) - 0.5) * h_fov * 1.05)
        bottom = math.sin(math.pi / 2 - pitch - ((y / float(sc_height)) * v_fov) + v_fov/2)
        angle = math.atan(top / bottom)

        print "top (x) is: " + str(top) + ", bottom (y) is: " + str(bottom)
        print "width ratio is: " + str(x/sc_width) + ", x is: " + str(x)

        print "angle is: " + str(angle)

        if -math.radians(45) > angle or angle > math.radians(45):
            angle = 0

        if x/sc_width < 0.5 and angle > 0:
            angle = 0
        elif x/sc_width > 0.5 and angle < 0:
            angle = 0

        # calculate heading of target in absolute frame
        bearing = angle + heading

        if bearing < 0:
            bearing = 2 * math.pi + bearing
        elif bearing > 2 * math.pi:
            bearing = bearing - 2 * math.pi

        # check enough bearings within a range have been identified
        # before calculating and sending gps coordinates
        if len(self.t_bearings) > 10:
            del self.t_bearings[0]
        self.t_bearings.append(bearing)

        print 'target bearing is: ' + str(bearing)

        # degrees in radians
        bearing_thresh = math.radians(0.5)

        # set some thresholds to check previous target bearings are within
        upper = bearing + bearing_thresh
        lower = bearing - bearing_thresh

        thresh_count = 0
        for bearing in self.t_bearings:
            if lower < bearing < upper:
                thresh_count += 1

        if thresh_count >= 5:

            # define conversion ratios for degrees to metres
            lat_rad = math.radians(lat)
            deg_2_m_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * \
                math.cos(4 * lat_rad) - 0.0023 * math.cos(6 * lat_rad)
            deg_2_m_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad) + 0.118 * math.cos(5 * lat_rad)

            # difference in latitude and longitude from current aircraft location and point on bearing to fly toward
            length_n = length * math.cos(bearing)
            length_e = length * math.sin(bearing)
            delta_lat = length_n / deg_2_m_lat
            delta_lon = length_e / deg_2_m_lon

            # calculate gps coords of target using differences and plane gps position
            t_lat = lat + delta_lat
            t_lon = lon + delta_lon
            gps_coords = (t_lat, t_lon)

            return gps_coords

    def bearing_find_expand(self, ti_coord, sc_height, sc_width):

        # return if values haven't been assigned
        if ti_coord is None or sc_height is None or sc_width is None:
            return

        length = 300.0

        # check drone is level enough as it messes up calculations if not
        roll = self.v.attitude.roll
        thresh = math.radians(5.0)  # 5 degrees in radians
        if not thresh > roll > -thresh:
            return

        # get some values from the drone that are needed
        # to calculate bearing from current heading
        pitch = self.v.attitude.pitch * -1  # invert so positive pitch is pointed downward

        # get some values from the drone that are needed
        # to calculate gps coordinates to fly toward
        heading = math.radians(self.v.heading)
        lat = self.v.location.global_frame.lat
        lon = self.v.location.global_frame.lon

        # assign FOV angles and convert to radians
        v_fov = math.radians(21.6)  # 21.6 degrees in radians
        h_fov = math.radians(30.6)  # 30.6 degrees in radians

        # extract x and y coordinates of target in image
        x = float(ti_coord[0])
        y = float(ti_coord[1])



        print "angle is: " + str(angle)

        if -math.radians(45) > angle or angle > math.radians(45):
            angle = 0

        # calculate heading of target in absolute frame
        bearing = angle + heading

        if bearing < 0:
            bearing = 2 * math.pi + bearing
        elif bearing > 2 * math.pi:
            bearing = bearing - 2 * math.pi

        # check enough bearings within a range have been identified
        # before calculating and sending gps coordinates
        if len(self.t_bearings) > 10:
            del self.t_bearings[0]
        self.t_bearings.append(bearing)

        print 'target bearing is: ' + str(bearing)

        # degrees in radians
        bearing_thresh = math.radians(0.5)

        # set some thresholds to check previous target bearings are within
        upper = bearing + bearing_thresh
        lower = bearing - bearing_thresh

        thresh_count = 0
        for bearing in self.t_bearings:
            if lower < bearing < upper:
                thresh_count += 1

        if thresh_count >= 5:

            # define conversion ratios for degrees to metres
            lat_rad = math.radians(lat)
            deg_2_m_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * \
                math.cos(4 * lat_rad) - 0.0023 * math.cos(6 * lat_rad)
            deg_2_m_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad) + 0.118 * math.cos(5 * lat_rad)

            # difference in latitude and longitude from current aircraft location and point on bearing to fly toward
            length_n = length * math.cos(bearing)
            length_e = length * math.sin(bearing)
            delta_lat = length_n / deg_2_m_lat
            delta_lon = length_e / deg_2_m_lon

            # calculate gps coords of target using differences and plane gps position
            t_lat = lat + delta_lat
            t_lon = lon + delta_lon
            gps_coords = (t_lat, t_lon)

            return gps_coords