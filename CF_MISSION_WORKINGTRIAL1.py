## IMPORTS (same as found in bitcraze example code "logandfly.py", but instead of importing position commander we import motion commander)
import os  #for saving the log
import time #for timing
import datetime as dt   #for dating the log with timestamps
import numpy as np  #math
import cflib.crtp   #communication w/ drone
from cflib.crazyflie import Crazyflie   #import crazyflie class to instantiate the drone; self explanatory
from cflib.crazyflie.log import LogConfig #logging capabilities
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie #real-time communication with drone
from cflib.positioning.motion_commander import MotionCommander #velocity control, plus takeoff and landing (not using position commander)
from cflib.utils import uri_helper #connect to radio address
import warnings #used in main() to remove spam warnings from the terminal so we can actually read the printed output

## ------------ SETTINGS & PARAMETERS -----------------------------------------------------------

# Flight default settings
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7A7')# Radio address of the drone
CTRL_LOOP_DT = 0.03  #in seconds, control loop period (should be longer than logging period to ensure data is fresh)
LOG_DT = 0.02 # in seconds, time period between log samples 
HEIGHT = 0.5 #in m, default height of the drone
SPEED = 0.3 # m/s, default horizontal speed
ASCENT_SPEED = 0.2 # m/s, takeoff speed in z (away from ground)
DESCENT_SPEED = 1.0 # m/s, landing speed in z (fast, so the drone barely moves sideways on the way down)
SETTLING_TIME = 1.0  # s, settling time after a move so the drone stops drifting before we attempt to land (for example)
REACHED_TOL = 0.05 # m, how close the drone must be to the target to count as arrived (in real life, the drone will never reach an exact position)

# Mission-specific settings
# The search covers LANDING_SEARCH_RANGE_X by LANDING_SEARCH_RANGE_Y with a center of the guess
EST_LANDING_X = 3.8 # m, rough landing pad position away from the takeoff/home position
EST_LANDING_Y = 0.25 # m, rough landing pad position to the left of the takeoff pad
LANDING_SEARCH_RANGE_X = 0.7  # m, size of the search area along x
LANDING_SEARCH_RANGE_Y = 1 # m, same but along y 
HOME_SEARCH_RANGE_XY = 0.4 # m, this is the bounding box side length for the return search (we choose a square for simplicity since we know this position better)
MAX_LANE_WIDTH = 0.1  # m, largest allowed gap between lanes (must be smaller than box or risks missing)
BOX_DIP = 0.08 # m, threshold value for detecting a dip in the downward z sensor
EMPTY_LANES_B4_STOP = 2  # after the box is seen, stop after this many lanes with no box detected

# Obstacle avoidance parameters
# The drone only ever moves along one axis at a time, so the actual obstacle sensing problem is 1D (see move_xy aglorithm for details)
# If something is ahead, it slides sideways until the way ahead is clear (plus a margin), then carries on forward.
OBST_THRESHOLD_DIST = 0.25  # m, so an obstacle closer than this along the motion axis ahead triggers a dodge 
OBST_CLEARED_DIST = OBST_THRESHOLD_DIST + 0.1   # m, distance where an obstacle is assumed to no longer be in front (slightly above threshold to avoid back-and-forth oscillations)
DRONE_HITBOX = 0.12 # m, additional dodge (divided by 2) after the way ahead clears to make sure drone passes the obstacle
FAR = 4.0 # m, distance used for a null/invalid reading from multiranger; distance from cf multiranger spec

## ------- LOGGING CAPABILITY -----------------------------------------------------------------

# Logging format setup: Crazyflie variable, variable type, header name
# range is in millimeters, stateEstimate (position) is in meters
#this was made into its own list because it gets used a few times later and is easier to just write it out once
LOG_VARS = [('range.front',     'uint16_t', 'front_mm'),
('range.back',      'uint16_t', 'back_mm'),
('range.up',        'uint16_t', 'up_mm'),
('range.left',      'uint16_t', 'left_mm'),
('range.right',     'uint16_t', 'right_mm'),
('stateEstimate.x', 'float',    'x_m'),
('stateEstimate.y', 'float',    'y_m'),
('stateEstimate.z', 'float',    'z_m'),
('range.zrange',    'uint16_t', 'down_mm')]    # down sensor (very important for box detection)

# Position (column) of each value in a log row (matches above)
FRONT_COL = 0
BACK_COL = 1
UP_COL = 2
LEFT_COL = 3
RIGHT_COL = 4
X_COL = 5
Y_COL = 6
Z_COL = 7
DOWN_COL = 8

#This function tells the drone to send back the data we want to log, and then saves it to the log list (named "rows")
# Note: The drone MUST be connected first
def start_logging(scf, rows): #setup the log configuration 
    lg = LogConfig(name='Mission', period_in_ms=int(LOG_DT * 1000))  #define samples per second (hz) found as 1/period (in ms)
    for var, var_type, _ in LOG_VARS: #We are just taking our earlier setup of the log variables and using those (ignoring the header name via _)
        lg.add_variable(var, var_type) #tell the drone to log these specific variables (with types defined)

    def on_data(timestamp, data, _): #this function is called every time a new sample is received on the computer from the drone
        rows.append([data[var] for var, _, _ in LOG_VARS]) #simply add the new data to the list
 
    scf.cf.log.add_config(lg) #send the config to the drone
    lg.data_received_cb.add_callback(on_data)   #tell the computer to run on_data() when a sample is received
    lg.start() #start the logging (drone sends data to computer, computer runs on_data() when received)
    return lg #Return the log config so that it can be stopped later in the main loop using lg.stop()

def establish_data_connection(rows): #need to run this after start_logging() to make sure we actually receive data
    t0 = time.time() #set current time
    while not rows: #complete loop until rows is no longer empty
        if time.time() - t0 > 5.0: #watchdog to catch if data is not being sent (give up after 5 s)
            raise RuntimeError('Logging not working... try again') 
        time.sleep(CTRL_LOOP_DT) #delay for one control cycle to see if the drone will send data in the next

def get_pos_xyz(rows): #reads the most recent position data from the log and outputs it as an array of[x, y, z]
    row = rows[-1] #fetch the most recent data
    return np.array([row[X_COL], row[Y_COL], row[Z_COL]]) #extract xyz and return as array

def save_log(rows): #Function to save the log, used at the end of the main loop. Self explanatory
    os.makedirs('logs', exist_ok=True) #make log folder if it doesnt already exist
    name = dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv') #name it the mission time
    header = ','.join(col for _, _, col in LOG_VARS) #headers
    np.savetxt(os.path.join('logs', name), np.array(rows), delimiter=',', header=header, comments='') #save

## ---------------- BASIC MOTION COMMAND FUNCTIONS ----------------------------------------------------

def move_xy(mc, rows, x, y):
#the basic idea is that this command will fly the drone to a target xy while avoiding obstacles
#This is achieved by first completing an "x" move where the drone will fly to the target x position. 
#If it encounters an obstacle along the x direction, it will dodge in the y direction until the x direction is clear. 
# It repeats until the x position is reached. Then it does the same process for the y direction. 
# It iterates until the final position is reached within tolerance

# ------ HELPER FUNCTIONS FOR move_xy() ---------
    def get_sens_dist(move_axis, sign): #we send the direction as the arg and receive the obstacle distance in meters
        #move_axis: 0 is x and 1 is y. sign: 1 is positive and -1 is negative. Ex: axis 0, sign 1 is the positive x direction (front sensor)
        if move_axis == 0:
            col = FRONT_COL if sign > 0 else BACK_COL #look front or back if motion is in x axis
        else:
            col = LEFT_COL if sign > 0 else RIGHT_COL #look R/L for motion in y axis
        d = rows[-1][col] / 1000.0 #select the most recent data point for the desired position data, and find in meters
        #we will assume that if the reading is <= 0 that the reading is invalid and means the sensor is essentially free in that direction (thus "FAR")
        return d if d > 0.0 else FAR #if real data is retrieved give back the data

    def cmd_velocity(move_axis, axis_vel): #command the drone to move with a specified velocity in the xy plane
        #drone will move with specified velocity until told otherwise (either via a new call of this function or mc.stop())
        v = [0.0, 0.0] #initialize the velocity array
        v[move_axis] = axis_vel #set the commanded axis velocity as specified in the input (- means backwards)
        mc.start_linear_motion(v[0], v[1], 0.0) #args are x, y, z velocities. We only want to move in x and y

    #This function tells the drone to move to the side incrementally until the path forward is clear enough to proceed
    def dodge(move_axis, sign): #arguments are the current motion axis and direction (+/-)
        side_ax = 1 - move_axis #if motion axis is 0 (x) then the side dir is 1 (y), and if motion axis is y then the side dir is x
        move_dir = 1 if get_sens_dist(side_ax, 1) > get_sens_dist(side_ax, -1) else -1 #choose to move to the side with more room (if equal it moves negatively, arbitrarily)
        dodged_dist = 0.0    # sideways distance covered since the way ahead cleared (initialized to 0, because motion has not yet started)
        while dodged_dist < DRONE_HITBOX/2: #keep going until the drone reaches the defined dodging distance
            #DRONE_HITBOX/2 is the distance we want to move to the side in addition to the distance needed to satisfy the safe distance requirement
            cmd_velocity(side_ax, move_dir * SPEED) #tell drone to move to the side with the given speed in the chosen side direction
            if get_sens_dist(move_axis, sign) >= OBST_CLEARED_DIST: #If we cleared the obstacle (within the OBST_CLEARED_DIST threshold) begin to accumulate dodged distance for extra buffer
                dodged_dist += SPEED * CTRL_LOOP_DT #new distance = current dist + speed*time
                print('finishing dodge')
            else:
                dodged_dist = 0.0 #If the obstacle is still detected ahead (within the threshold) we have not yet dodged, so the dist will remain 0
                print('obstacle ahead - dodging')
            #IMPORTANT: we must allow the loop to sleep. If we do not allow this, our speed*CTRL_LOOP_DT will accumulate too fast. This is also good to ensure sensor data is fresh and we don't spam commands to the drone
            time.sleep(CTRL_LOOP_DT) 

    #Function that combines cmd_velocity and dodging to complete a full linear motion
    def move_1d(move_axis, axis_vel):
        while True: #repeat until this axis is within tolerance (the break below ends the loop)
            #compute error between the target position along the direction of interest (argument of the parent function) and the measured position on that axis (from optical flow sensor)
            err = target[move_axis] - get_pos_xyz(rows)[move_axis] #desired - actual
            #Basic idea is that if the error is within the tolerance, the move is completed. If error is above that, we continue
            if abs(err) < REACHED_TOL:
                break
            sign = 1 if err > 0 else -1 #if error is positive we are behind the target so move "forward"; if negative we need to reverse direction
            if get_sens_dist(move_axis, sign)  < OBST_THRESHOLD_DIST:  #if an obstacle is found in the motion direction, perform a dodge
                dodge(move_axis, sign)
            else:
                cmd_velocity(move_axis, sign * axis_vel) #otherwise, move towards the target along the direction of interest
            time.sleep(CTRL_LOOP_DT)

    target = [x, y] #set the goal coords in an array

    #Move to defined target point while avoiding obstacles
    [x_read, y_read, _] = get_pos_xyz(rows) #take initial reading
    current_ax = 0   #define first motion direction (will always be x; arbitrary convention)
    final_err =np.sqrt((x_read - x)**2 + (y_read - y)**2) #find radial error between current pos and desired pos
    while final_err > np.sqrt(2)*REACHED_TOL: #while error is greater than desired tolerance zone (with correction factor to account for cirlular geometry vs square) perform the loop
        move_1d(current_ax , SPEED) #move along desired axis, dodging obstacles along the way
        [x_read, y_read, _] =get_pos_xyz(rows) #take pos again
        final_err = np.sqrt((x_read - x)**2 + (y_read - y)**2) #find error again
        
        #change motion axis by alternating between 0 and 1
        current_ax = abs(current_ax - 1)

        #repeat alternating between x and y motions until target is reached
    mc.stop() # hover command (once position is reached)
    time.sleep(SETTLING_TIME) #allow drone to settle in place

#land and takeoff functions... self explanatory
def take_off(mc):
    mc.take_off(HEIGHT, ASCENT_SPEED) #just commanding takeoff using motioncommander (note that this sets the current pos to 0, 0)

def land(mc):
    mc.land(DESCENT_SPEED) #use motion commander to land, with specified descent speed

## SEARCH ALGORITHM --- MAIN CHALLENGE
#this function takes inputs of the log object (which is constantly updating) and the desired range of the search region
# It performs a zigzag/raster scan as the search pattern
# During the motion it collects the log data of the xy locations of the points where a box edge is detected with the height sensor
# The centroid of the detected box edge points is found and returned as the box center position
# If the box is found, the search ends without completing fully in order to save time (for battery life)
def land_find(mc, rows, x_range, y_range): #this function performs a search over a rectangle area for the landing pad, and returns the xy location result from the search
    x_min, x_max = x_range #x bounds of the search
    y_min, y_max = y_range #y bounds of the search
    # Lanes sit on both edges of the area, so n lanes leave (n - 1) equal gaps between them
    # Start with 2 lanes (one on each edge) and add lanes until the gap is no more than MAX_LANE_WIDTH
    lane_count = 2
    lane_spacing = (x_max - x_min) / (lane_count - 1)    # m, the actual gap between lanes (initialize for first pass)
    while lane_spacing > MAX_LANE_WIDTH: #while the current lane spacing is more than the allowable:
        lane_count += 1 #increase # of lanes in the search (reducing lane spacing...) until spacing is below the threshold
        lane_spacing = (x_max - x_min) / (lane_count - 1) #update lane spacing

    #Now we want to get a baseline reading for the z height (used later to detect the dip)
    log = np.array(rows)    #get the log as a numpy array (easiest to work with for next line)
    recent_z_reads = range(len(log) - 10, len(log))    # row numbers of the last 10 samples (about 0.4 s of hovering)
    floor_h = np.mean(log[recent_z_reads, DOWN_COL]) / 1000.0    # avg of down readings in m to get baseline floor height estimate

    edge_coords = [] #initialize array of xy coords of every sample over the box
    empty_rows = 0  # lanes with no box since the box was last seen (this is used to end the search once it is clear the box has been found)

    #Now command the actual search to start
    for i in range(lane_count): #perform the strategy for the specified number of lanes (found just prior)
        lane_x = x_min + i * lane_spacing #fidn the y pos of the lane that we want to command
        print('Starting search along lane in y at x of', lane_x)
        if i % 2 == 0: #if index is evenly divisible by 2 (ie i/2 is an integer) then it is an even number (using mod operator %)
            y_start, y_end = (y_min, y_max) #even lane moves from min to max (essentially forward) (note that index starts at 0, not 1, so the first lane is even)
        else:
            y_start, y_end = (y_max, y_min) #odd lanes move backwards

        initial_row_i = len(rows) #index of the log data prior to start the search
        move_xy(mc, rows, lane_x, y_start) #move to the starting position for this lane
        move_xy(mc, rows, lane_x, y_end) #perform the line move down the lane
        lane_rows = rows[initial_row_i:len(rows)] #collect all the data for xyz (and others) that was generated during the lane sweep

        #intialize a variable that counts how many dips we found during the lane pass
        dip_num = 0
        #now we will go through and check the data from the pass to see if there were any dips (and if so, we will log the xy position)
        for r in lane_rows: #check all of the data we just collected
            down_meas = r[DOWN_COL] / 1000.0 #report the measurement collected by the downward looking z sensor
            #if the difference between our current measurement and expected floor measurement is greater than some threshold (related to box height) log it as a datapoint
            if abs(floor_h - down_meas) >= BOX_DIP: #use abs to catch both the beginning and end dips on this pass
                edge_coords.append((r[X_COL], r[Y_COL])) #we only care about the xy pos. Append these to our list of box points
                dip_num += 1 #acknowledge that we found dips (used for the next part)

        #if we go for a certain # of lanes without finding dips after some of already been detected, this means we have found the box and can end the search early
        if dip_num > 0: #dips found... we clearly need to keep searching because there might be more!
            empty_rows = 0 #tells us we have not yet found any empyt rows 
        elif edge_coords: #if box points is not empty (meaning prior points have been detected)
            empty_rows += 1 #now we have clearly found an empty row that is occuring after the box
            if empty_rows >= EMPTY_LANES_B4_STOP: #if we go above our set threshold (say 1-2 rows), end the search
                break    # we have passed the box

    if not edge_coords: #important to catch the edge case when no box is found
        print('Search did not find a valid landing zone') #quick diagnostic message
        return None

    #now we find the estimated box center position by taking the mean of all the data points we found in each direction (in theory, errors should average out assuming we got enough data points)
    centroid_x = np.mean([x for x, _ in edge_coords]) #find x center (ignore y data)
    centroid_y = np.mean([y for _, y in edge_coords]) #find y center (ignore x data)
    print('Box found at approximate xy of ', centroid_x, centroid_y, 'relative to takeoff point') #let user know the box was found (sanity check)
    return centroid_x, centroid_y #return the box position

## ----- GENERAL MISSION ALGORITHM -----------------------
#This function defines the full mission (strictly from an algorithmic perspective)
#It is broken into eight distinct stages (takeoff, move to land 1, find pad, land, takeoff again, move back to home, find home pad, land)
# Obstacle avoidance logic and search method logic are found in their respective functions... this is strictly high level implementation

def mission(mc, rows):
    landing_range_x = (EST_LANDING_X - LANDING_SEARCH_RANGE_X / 2 , EST_LANDING_X + LANDING_SEARCH_RANGE_X / 2 )
    landing_range_y = (EST_LANDING_Y - LANDING_SEARCH_RANGE_Y / 2 , EST_LANDING_Y + LANDING_SEARCH_RANGE_Y / 2) 
    #First phase of the mission
    take_off(mc)    # from the takeoff pad (this also sets the current xy as 0, 0)
    time.sleep(SETTLING_TIME) #allow drone to settle
    home_x, home_y = 0.0, 0.0 #set current xy as 0,0 (home). This is just for bookkeeping

    #Second part of the mission: fly from the takeoff pad to the start of the search region
    move_xy(mc, rows, landing_range_x[0], landing_range_y[0])
    print('starting search')

    #Third part of the mission: perform the search to find the landing pad coordinates
    land_pad_coords = None #nothing found yet
    #land_counter = 0
    land_pad_coords = land_find(mc, rows, landing_range_x, landing_range_y)    # (x, y) of the landing pad, or None
    #Below is an idea to handle the case when the box is not found, but it was buggy
    #while land_pad_coords is None: #if we do not find the landing pad on the first search, we try again (while slightly expanding the search range)
        #landing_range_x = (EST_LANDING_X - LANDING_SEARCH_RANGE_X / 2 - 0.1*land_counter, EST_LANDING_X + LANDING_SEARCH_RANGE_X / 2 + 0.1*land_counter)
        #landing_range_y = (EST_LANDING_Y - LANDING_SEARCH_RANGE_Y / 2 -0.1*land_counter, EST_LANDING_Y + LANDING_SEARCH_RANGE_Y / 2 + 0.1*land_counter) 
        #land_pad_coords = land_find(mc, rows, landing_range_x, landing_range_y)    # (x, y) of the landing pad, or None
        #land_counter += 1 # increase counter
    print('found pad')

    #Fourth part: perform the landing maneuver
    move_xy(mc, rows, land_pad_coords[0], land_pad_coords[1]) #move to position over the landing pad
    land(mc) #do the landing
    pos = get_pos_xyz(rows) # estimate of where we actually landed (useful for the next stage of the mission)
    time.sleep(5) # allow the drone to rest for a moment before performing the takeoff

    #Fifth part: take off from the landing pad
    take_off(mc) #Keep in mind that the takeoff resets our home position (this is based on how the drone logs things)
    home_x, home_y = -pos[0], -pos[1] #translating our original coordinates into our new reference frame

    #Sixth part: return to our home position (avoiding obstacles along the way)
    move_xy(mc, rows, home_x, home_y)

    #Seventh part: find the position of the takeoff pad (in case there was drift)
    #we provide a separate range for the search here since the position is known now with much more accuracy than the original landing pad pos was
    home_pos = None #we do not know with certainty where the box is yet so we set this value to none
    home_counter = 0 #same logic as was used for the original landing
    while home_pos is None: #if we do not find the takeoff pad on the first search, we try again (while slightly expanding the search range). Same as we did for the first landing
        home_pos = land_find(mc, rows, (home_x - HOME_SEARCH_RANGE_XY - 0.1*home_counter, home_x + HOME_SEARCH_RANGE_XY + 0.1*home_counter), (home_y - HOME_SEARCH_RANGE_XY - 0.1*home_counter, home_y + HOME_SEARCH_RANGE_XY + 0.1*home_counter))
        home_counter += 1 # increase counter

    #Eighth (FINAL) part: perform the landing on the original home pad
    move_xy(mc, rows, home_pos[0], home_pos[1]) #move to over the pad
    land(mc) #LAND!!! (wooooo... hopefully)

## ------ MAIN LOOP ----------------
#although the actual mission logic is defined above, this loop handles the final high-level details
#This includes setting up the drivers, log list, connection with cf, and ensuring landing in case of errors (and saving the log)

def main_loop():
    cflib.crtp.init_drivers() #initialize low-level drivers to connect to cf
    rows = [] #initialize the log data list

    #connect to cf (mirrors bitcraze examples)
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        lg = start_logging(scf, rows) #begin logging

        #first make sure we can actually receive data from the drone and setup the motioncommander (log data always gets saved at the end even if an issue happens)
        try:
            establish_data_connection(rows)
            mc = MotionCommander(scf, default_height=HEIGHT)
            t0 = time.time() #get mission start time (this is in the log but is convenient here)

            #if the above executed properly, we can now fly the mission (yay)! The drone always lands even if something goes wrong
            try:
                mission(mc, rows) #This is the whole point lol
            finally:
                land(mc) # For drone safety: lands if the mission stopped early (error/Ctrl+C) and does nothing if already landed
                print('Mission time was: ', time.time() - t0, ' seconds') #can be found in logs, but good diagnostic to make easy to read upfront
        finally: #always make sure we stop the logging and save the log data (VERY IMPORTANT)
            lg.stop()
            save_log(rows)

#Run the main loop if this file is executed (self explanatory)
if __name__ == '__main__':
    warnings.filterwarnings("ignore") 
    main_loop()