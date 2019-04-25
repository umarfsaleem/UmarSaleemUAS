from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math


class Autopilot:

    def __init__(self, height, width):
        print ('autopilot communication started')

        self.connection_string = '/dev/ttyS0'
        self.v = connect(self.connection_string, wait_ready=True, baud=57600)
        self.start_checkpoint = 1
        self.end_checkpoint = 4
        self.sc_height = height
        self.sc_width = width
        self.none_start = 0.0
        self.t_bearings = []

        # assign FOV angles and convert to radians
        self.v_fov = math.radians(44.46)  # vertical fov
        self.h_fov = math.radians(62.76)  # horizontal fov

        self.wait_to_arm()

        # define conversion ratios for degrees to metres
        lat_rad = math.radians(self.v.location.global_frame.lat)
        self.deg_2_m_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * \
                           math.cos(4 * lat_rad) - 0.0023 * math.cos(6 * lat_rad)
        self.deg_2_m_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad) + 0.118 * math.cos(5 * lat_rad)

    def direct(self, coordinates):
        if self.safety_check():
            # safe to direct
            if coordinates is not False:
                # target has been identified
                self.none_start = time.time()

                target_gps_coords = self.bearing_find(coordinates)

                if self.v.mode.name is not 'GUIDED':
                    self.v.mode = VehicleMode("GUIDED")

                if target_gps_coords is not False:
                    lat = target_gps_coords[0]
                    lon = target_gps_coords[1]
                    altitude = self.v.location.global_frame.alt
                    self.v.simple_goto(LocationGlobalRelative(lat, lon, altitude))
                    return target_gps_coords
                else:
                    return False

            else:
                # no target was found
                none_time = time.time() - self.none_start
                if none_time > 15 and self.v.mode.name is not 'AUTO':
                    self.v.mode = VehicleMode("AUTO")
                return False
        else:
            return False

    def safety_check(self):
        if not self.v.armed:
            self.wait_to_arm()
    
        if self.v.mode.name is not 'AUTO' and self.v.mode.name is not 'GUIDED':
            return False
    
        if self.v.commands.next <= self.start_checkpoint:
            return False
    
        if self.v.commands.next > self.end_checkpoint:
            return False
    
        return True
    
    def wait_to_arm(self):
        already_alerted = False
        while not self.v.armed:
            if not already_alerted:
                print('waiting to arm...')
                already_alerted = True
            time.sleep(1)
    
        print('Armed')
    
    def bearing_find(self, coordinates):
        length = 300.0
    
        roll = self.v.attitude.roll
        thresh = math.radians(5.0)  # 5 degrees in radians
        if not thresh > roll > -thresh:
            return False
    
        pitch = self.v.attitude.pitch * -1  # invert so positive pitch is pointed downward
    
        heading = math.radians(self.v.heading)
        lat = self.v.location.global_frame.lat
        lon = self.v.location.global_frame.lon
    
        # extract x and y coordinates of target in image
        x = float(coordinates[0])
        y = float(coordinates[1])
    
        # calculate relative angle of target from current aircraft position
        top = math.tan(((x / float(self.sc_width)) - 0.5) * self.h_fov * 1.05)
        bottom = math.sin(math.pi / 2 - pitch - ((y / float(self.sc_height)) * self.v_fov) + self.v_fov / 2)
        angle = math.atan(top / bottom)
    
        # discard angles outside 45 degrees either way
        if -math.radians(45) > angle or angle > math.radians(45):
            angle = 0
    
        # discard angles where left/right don't match
        # the side the target is in
        if x / self.sc_width < 0.5 and angle > 0:
            angle = 0
        elif x / self.sc_width > 0.5 and angle < 0:
            angle = 0
    
        # calculate heading of target in absolute frame and keep within
        # range of 0-360 degrees
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
        bearing_thresh = math.radians(0.5)
        upper = bearing + bearing_thresh
        lower = bearing - bearing_thresh
        thresh_count = 0
        for bearing in self.t_bearings:
            if lower < bearing < upper:
                thresh_count += 1
    
        if thresh_count >= 5:
            # difference in latitude and longitude from current aircraft location and point on bearing to fly toward
            length_n = length * math.cos(bearing)
            length_e = length * math.sin(bearing)
            delta_lat = length_n / self.deg_2_m_lat
            delta_lon = length_e / self.deg_2_m_lon
    
            # calculate gps coords of target using differences and plane gps position
            t_lat = lat + delta_lat
            t_lon = lon + delta_lon
            gps_coords = (t_lat, t_lon)
    
            return gps_coords
    
        else:
            return False
