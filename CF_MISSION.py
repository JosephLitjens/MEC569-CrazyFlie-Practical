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

## ------------ SETTINGS & PARAMETERS -----------------------------------------------------------

# Flight settings
CONTROL_DT = 0.05        # s, control loop period - should be longer than logging period
HEIGHT = 0.3             # m, working height
SPEED = 0.2              # m/s, horizontal speed
CLIMB_SPEED = 0.2        # m/s, takeoff speed in z
LAND_SPEED = 1.0         # m/s, landing speed in z (fast, so the drone barely moves sideways on the way down)
SETTLE_TIME = 1.0        # s, hover after a move so the drone stops drifting before we read its position
REACHED_TOL = 0.025       # m, how close the drone must be to the target to count as arrived (in real life, the drone will never reach an exact position)

# Mission settings
# Set PAD_GUESS to your rough estimate of the landing pad, measured from the takeoff pad, before each run.
# The search covers PAD_SEARCH_X by PAD_SEARCH_Y centered on that guess.
PAD_GUESS_X = 3.5        # m, rough landing pad position, forward of the takeoff pad
PAD_GUESS_Y = 0.0        # m, rough landing pad position, left of the takeoff pad (negative = right)
PAD_SEARCH_X = 0.5       # m, size of the search area along x
PAD_SEARCH_Y = 1.0       # m, size of the search area along y
HOME_SEARCH = 0.4        # m, on the way back, search this far each way around where the takeoff pad should be
MAX_LANE_SPACING = 0.1  # m, largest allowed gap between lanes (keep it smaller than the box)
BOX_DIP = 0.04           # m, a down reading this far below the floor = "over the box"
LOG_PERIOD = 0.04        # s between log samples 
EMPTY_LANES_TO_STOP = 2  # after the box is seen, stop after this many lanes with no box

# Obstacle avoidance settings
# The drone only ever moves along ONE axis at a time, so the actual obstacle sensing problem is 1D (see fly_to aglorithm for details)
# If something is ahead, it slides sideways until the way ahead is clear (plus a margin), then carries on forward.
SAFE_DIST = 0.25         # m, an obstacle closer than this ahead triggers a sidestep
CLEAR_DIST = SAFE_DIST + 0.10    # m, "clear again" distance (the gap stops flickering)
DRONE_HITBOX = 0.1       # m, keep sliding this far (divided by 2) after the way ahead clears to make sure you pass the obstacle
FAR = 4.0               # m, distance used for "nothing in range / invalid reading"; from crazyflie spec

# Radio address of the drone
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7A7')

## ------- LOGGING CAPABILITY -----------------------------------------------------------------

# Logging format setup: Crazyflie variable, variable type, header name
# range is in millimeters, stateEstimate (position) is in meters
LOG_VARS = [
('range.front',     'uint16_t', 'front_mm'),
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

#This function tells the drone to send back the data we want to log, and then saves it to the list ("rows")
# Note: The drone MUST be connected first
def start_logging(scf, rows): 
    #setup the log configuration 
    lg = LogConfig(name='Mission', period_in_ms=int(LOG_PERIOD * 1000))  #define samples per second (hz) found as 1/period (in ms)
    for var, var_type, _ in LOG_VARS: #We are just taking our earlier setup of the log variables and using those (ignoring the header name via _)
        lg.add_variable(var, var_type) #tell the drone to log these specific variables (with types defined)

    def on_data(timestamp, data, _): #this function is called every time a new sample is received on the computer from the drone
        rows.append([data[var] for var, _, _ in LOG_VARS]) #simply add the new data to the list
 
    scf.cf.log.add_config(lg) #send the config to the drone
    lg.data_received_cb.add_callback(on_data)   #tell the computer to run on_data() when a sample is received
    lg.start() #start the logging (drone sends data to computer, computer runs on_data() when received)
    return lg #Return the log config so that it can be stopped later in the main loop using lg.stop()

def wait_for_data(rows): #need to run this after start_logging() to make sure we actually receive data
    t0 = time.time() #set current time
    while not rows: #complete loop until rows is no longer empty
        if time.time() - t0 > 5.0: #watchdog to catch if data is not being sent (give up after 5 s)
            raise RuntimeError('No log data received from the Crazyflie')
        time.sleep(CONTROL_DT) #delay for one control cycle to see if the drone will send data in the next

def get_position(rows): #reads the most recent position data from the log and outputs it as an array of[x, y, z]
    row = rows[-1] #fetch the most recent data
    return np.array([row[X_COL], row[Y_COL], row[Z_COL]]) #extract xyz and return as array

def save_log(rows): #Function to save the log, used at the end of the main loop. Self explanatory
    os.makedirs('logs', exist_ok=True) 
    name = dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S.csv')
    header = ','.join(col for _, _, col in LOG_VARS)
    np.savetxt(os.path.join('logs', name), np.array(rows), delimiter=',', header=header, comments='')

## ---------------- BASIC MOTION COMMAND FUNCTIONS ----------------------------------------------------

def fly_to(mc, rows, x, y):
#the basic idea is that this command will fly the drone to a target xy while avoiding obstacles
#This is achieved by first completing an "x" move where the drone will fly to the target x position. 
#If it encounters an obstacle along the x direction, it will sidestep in the y direction until the x direction is clear. 
# It repeats until the x position is reached. Then it does the same process for the y direction. 
# It iterates until the final position is reached within tolerance

# ------ HELPER FUNCTIONS FOR fly_to() ---------
    def get_dist_m(axis, sign): #we send the direction as the arg and receive the obstacle distance in meters
        #axis: 0 is x and 1 is y. sign: 1 is positive and -1 is negative. Ex: axis 0, sign 1 is the positive x direction (front sensor)
        if axis == 0:
            col = FRONT_COL if sign > 0 else BACK_COL #look front or back if motion is in x axis
        else:
            col = LEFT_COL if sign > 0 else RIGHT_COL #look R/L for motion in y axis
        d = rows[-1][col] / 1000.0 #select the most recent data point for the desired position data, and find in meters
        #we will assume that if the reading is <= 0 that the reading is invalid and means the sensor is essentially free in that direction (thus "FAR")
        return d if d > 0.0 else FAR #if real data is retrieved give back the data

    def set_speed(axis, axis_vel): #command the drone to move with a specified velocity in the xy plane
        #drone will move with specified velocity until told otherwise (either via a new call of this function or mc.stop())
        v = [0.0, 0.0] #initialize the velocity array
        v[axis] = axis_vel #set the commanded axis velocity as specified in the input (- means backwards)
        mc.start_linear_motion(v[0], v[1], 0.0) #args are x, y, z velocities. We only want to move in x and y

    #This function tells the drone to move to the side incrementally until the path forward is clear enough to proceed
    def sidestep(axis, sign): #arguments are the current motion axis and direction (+/-)
        side_ax = 1 - axis #if motion axis is 0 (x) then the side dir is 1 (y), and if motion axis is y then the side dir is x
        move_dir = 1 if get_dist_m(side_ax, 1) > get_dist_m(side_ax, -1) else -1 #choose to move to the side with more room (if equal it moves negatively, arbitrarily)
        sidestepped_dist = 0.0    # sideways distance covered since the way ahead cleared (initialized to 0, because motion has not yet started)
        while sidestepped_dist < DRONE_HITBOX/2: #keep going until the drone reaches the defined sidestep distance
            #DRONE_HITBOX/2 is the distance we want to move to the side in addition to the distance needed to satisfy the safe distance requirement
            set_speed(side_ax, move_dir * SPEED) #tell drone to move to the side with the given speed in the chosen side direction
            if get_dist_m(axis, sign) >= CLEAR_DIST: #If we cleared the obstacle (within the CLEAR_DIST threshold) begin to accumulate sidestepped distance for extra buffer
                sidestepped_dist += SPEED * CONTROL_DT #new distance = current dist + speed*time
            else:
                sidestepped_dist = 0.0 #If the obstacle is still detected ahead (within the threshold) we have not yet sidestepped, so the dist will remain 0
            #IMPORTANT: we must allow the loop to sleep. If we do not allow this, our speed*control_DT will accumulate too fast. This is also good to ensure sensor data is fresh and we don't spam commands to the drone
            time.sleep(CONTROL_DT) 

    #Function that combines set_speed and sidestep to complete a full linear motion
    def move_along(axis, axis_vel):
        while True: #repeat until this axis is within tolerance (the break below ends the loop)
            #compute error between the target position along the direction of interest (argument of the parent function) and the measured position on that axis (from optical flow sensor)
            err = target[axis] - get_position(rows)[axis] #desired - actual
            #Basic idea is that if the error is within the tolerance, the move is completed. If error is above that, we continue
            if abs(err) < REACHED_TOL:
                break
            sign = 1 if err > 0 else -1 #if error is positive we are behind the target so move "forward"; if negative we need to reverse direction
            if get_dist_m(axis, sign)  < SAFE_DIST or get_dist_m(abs(axis-1), 1)  < SAFE_DIST or get_dist_m(abs(axis-1), -1) < SAFE_DIST: #if an obstacle is found in the motion direction, or to the side (within the threshold), perform the sidestep
                sidestep(axis, sign)
            else:
                set_speed(axis, sign * axis_vel) #otherwise, move towards the target along the direction of interest
            time.sleep(CONTROL_DT)

    target = [x, y] #set the goal coords in an array

    #Move to defined target point while avoiding obstacles
    [x_read, y_read, _] =get_position(rows) #take initial reading
    current_ax=0    #define first motion direction (will always be x; arbitrary convention)
    final_err=np.sqrt((x_read - x)**2 + (y_read - y)**2) #find radial error between current pos and desired pos
    while final_err > np.sqrt(2)*REACHED_TOL: #while error is greater than desired tolerance zone (with correction factor to account for cirlular geometry vs square) perform the loop
        move_along(current_ax, SPEED) #move along desired axis, sidestepping obstacles along the way
        [x_read, y_read, _] =get_position(rows) #take pos again
        final_err=np.sqrt((x_read - x)**2 + (y_read - y)**2) #find error again
        
        #change motion axis by alternating between 0 and 1
        current_ax=abs(current_ax-1)

        #repeat alternating between x and y motions until target is reached
    mc.stop()    # hover command (once position is reached)
    time.sleep(SETTLE_TIME) #allow drone to settle

#land and takeoff functions... self explanatory
def take_off(mc):
    mc.take_off(HEIGHT, CLIMB_SPEED) #just commanding takeoff using motioncommander (note that this sets the current pos to 0, 0)

def land(mc):
    mc.land(LAND_SPEED) #use motion commander to land, with specified descent speed

## SEARCH ALGORITHM --- MAIN CHALLENGE
#this function takes inputs of the log (which is constantly updating) and the desired range of the search region
# It performs a "lawn mower" pattern motion as the search pattern
# During the motion it collects the log data of the xy locations of the points where a box edge is detected with the height sensor
# The centroid of the detected box edge points is found and returned as the box center position
# If the box is found, the search ends without completing fully in order to save time (for battery life)
def land_find(mc, rows, x_range, y_range): #this function performs a search over a rectangle area for the landing pad, and returns the xy location result from the search
    x_min, x_max = x_range #x bounds of the search
    y_min, y_max = y_range #y bounds of the search
    # Lanes sit on both edges of the area, so n lanes leave (n - 1) equal gaps between them
    # Start with 2 lanes (one on each edge) and add lanes until the gap is no more than MAX_LANE_SPACING
    lane_count = 2
    lane_spacing = (y_max - y_min) / (lane_count - 1)    # m, the actual gap between lanes (initialize for first pass)
    while lane_spacing > MAX_LANE_SPACING: #while the current lane spacing is more than the allowable:
        lane_count += 1 #increase # of lanes in the search (reducing lane spacing...) until spacing is below the threshold
        lane_spacing = (y_max - y_min) / (lane_count - 1) #update lane spacing

    #Now we want to get a baseline reading for the z height (used later to detect the dip)
    log = np.array(rows)    #get the log as a numpy array (easiest to work with for next line)
    recent_z_reads = range(len(log) - 10, len(log))    # row numbers of the last 10 samples (about 0.4 s of hovering)
    floor = np.mean(log[recent_z_reads, DOWN_COL]) / 1000.0    # avg of down readings in m to get baseline floor height estimate

    box_points = [] #initialize array of xy coords of every sample over the box
    empty_rows = 0  # lanes with no box since the box was last seen (this is used to end the search once it is clear the box has been found)

    #Now command the actual search to start
    for i in range(lane_count): #perform the strategy for the specified number of lanes (found just prior)
        lane_y = y_min + i * lane_spacing #fidn the y pos of the lane that we want to command

        if i % 2 == 0: #if index is evenly divisible by 2 (ie i/2 is an integer) then it is an even number (using mod operator %)
            x_start, x_end = (x_min, x_max) #even lane moves from min to max (essentially forward) (note that index starts at 0, not 1, so the first lane is even)
        else:
            x_start, x_end = (x_max, x_min) #odd lanes move backwards

        first = len(rows) #index of the log data prior to start the search
        fly_to(mc, rows, x_start, lane_y) #move to the starting position for this lane
        fly_to(mc, rows, x_end, lane_y) #perform the line move down the lane
        lane_rows = rows[first:len(rows)] #collect all the data for xyz (and others) that was generated during the lane sweep

        #intialize a variable that counts how many dips we found during the lane pass
        found_dips = 0
        #now we will go through and check the data from the pass to see if there were any dips (and if so, we will log the xy position)
        for r in lane_rows: #check all of the data we just collected
            down_meas = r[DOWN_COL] / 1000.0 #report the measurement collected by the downward looking z sensor
            #if the difference between our current measurement and expected floor measurement is greater than some threshold (related to box height) log it as a datapoint
            if abs(floor - down_meas) >= BOX_DIP: #use abs to catch both the beginning and end dips on this pass
                box_points.append((r[X_COL], r[Y_COL])) #we only care about the xy pos. Append these to our list of box points
                found_dips += 1 #acknowledge that we found dips (used for the next part)

        #if we go for a certain # of lanes without finding dips after some of already been detected, this means we have found the box and can end the search early
        if found_dips>0: #dips found... we clearly need to keep searching because there might be more!
            empty_rows = 0 #tells us we have not yet found any empyt rows 
        elif box_points: #if box points is not empty (meaning prior points have been detected)
            empty_rows += 1 #now we have clearly found an empty row that is occuring after the box
            if empty_rows >= EMPTY_LANES_TO_STOP: #if we go above our set threshold (say 1-2 rows), end the search
                break    # we have passed the box

    if not box_points: #important to catch the edge case when no box is found
        print('Search did not find a valid landing zone') #quick diagnostic message
        return None

    #now we find the estimated box center position by taking the mean of all the data points we found in each direction (in theory, errors should average out assuming we got enough data points)
    center_x = np.mean([x for x, _ in box_points]) #find x center (ignore y data)
    center_y = np.mean([y for _, y in box_points]) #find y center (ignore x data)
    print('Box found at approximate xy of ', center_x, center_y, 'relative to takeoff point') #let user know the box was found (good sanity check)

    return center_x, center_y #return the box position

## ----- GENERAL MISSION ALGORITHM -----------------------
#This function defines the full mission (strictly from an algorithmic perspective)
#It is broken into eight distinct stages (takeoff, move to land 1, find pad, land, takeoff again, move back to home, find home pad, land)
# Obstacle avoidance logic and search method logic are found in their respective functions... this is strictly high level implementation

def fly_mission(mc, rows):
    pad_x_range = (PAD_GUESS_X - PAD_SEARCH_X / 2 , PAD_GUESS_X + PAD_SEARCH_X / 2 )
    pad_y_range = (PAD_GUESS_Y - PAD_SEARCH_Y / 2 , PAD_GUESS_Y + PAD_SEARCH_Y / 2) 
    #First phase of the mission
    take_off(mc)    # from the takeoff pad (this also sets the current xy as 0, 0)
    time.sleep(SETTLE_TIME) #allow drone to settle
    home_x, home_y = 0.0, 0.0 #set current xy as 0,0 (home). This is just for bookkeeping

    #Second part of the mission: fly from the takeoff pad to the start of the search region
    fly_to(mc, rows, pad_x_range[0], pad_y_range[0])

    #Third part of the mission: perform the search to find the landing pad coordinates
    land_pad_coords = None #nothing found yet
    land_counter = 0
    while land_pad_coords is None: #if we do not find the landing pad on the first search, we try again (while slightly expanding the search range)
        pad_x_range = (PAD_GUESS_X - PAD_SEARCH_X / 2 - 0.1*land_counter, PAD_GUESS_X + PAD_SEARCH_X / 2 + 0.1*land_counter)
        pad_y_range = (PAD_GUESS_Y - PAD_SEARCH_Y / 2 -0.1*land_counter, PAD_GUESS_Y + PAD_SEARCH_Y / 2 + 0.1*land_counter) 
        land_pad_coords = land_find(mc, rows, pad_x_range, pad_y_range)    # (x, y) of the landing pad, or None
        land_counter += 1 # increase counter

    #Fourth part: perform the landing maneuver
    fly_to(mc, rows, land_pad_coords[0], land_pad_coords[1]) #move to position over the landing pad
    land(mc) #do the landing
    pos = get_position(rows) # estimate of where we actually landed (useful for the next stage of the mission)
    time.sleep(5) # allow the drone to rest for a moment before performing the takeoff

    #Fifth part: take off from the landing pad
    take_off(mc) #Keep in mind that the takeoff resets our home position (this is based on how the drone logs things)
    home_x, home_y = -pos[0], -pos[1] #translating our original coordinates into our new reference frame

    #Sixth part: return to our home position (avoiding obstacles along the way)
    fly_to(mc, rows, home_x, home_y)

    #Seventh part: find the position of the takeoff pad (in case there was drift)
    #we provide a separate range for the search here since the position is known now with much more accuracy than the original landing pad pos was
    home_pos = None #we do not know with certainty where the box is yet so we set this value to none
    home_counter = 0 #same logic as was used for the original landing
    while home_pos is None: #if we do not find the takeoff pad on the first search, we try again (while slightly expanding the search range). Same as we did for the first landing
        home_pos = land_find(mc, rows, (home_x - HOME_SEARCH - 0.1*home_counter, home_x + HOME_SEARCH + 0.1*home_counter), (home_y - HOME_SEARCH - 0.1*home_counter, home_y + HOME_SEARCH + 0.1*home_counter))
        home_counter += 1 # increase counter

    #Eighth (FINAL) part: perform the landing on the original home pad
    fly_to(mc, rows, home_pos[0], home_pos[1]) #move to over the pad
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
            wait_for_data(rows)
            mc = MotionCommander(scf, default_height=HEIGHT)
            t0 = time.time() #get mission start time (this is in the log but is convenient here)

            #if the above executed properly, we can now fly the mission (yay)! The drone always lands even if something goes wrong
            try:
                fly_mission(mc, rows) #This is the whole point lol
            finally:
                land(mc) # For drone safety: lands if the mission stopped early (error/Ctrl+C) and does nothing if already landed
                print('Mission time was: ', time.time() - t0, ' seconds') #can be found in logs, but good diagnostic to make easy to read upfront
        finally: #always make sure we stop the logging and save the log data (VERY IMPORTANT)
            lg.stop()
            save_log(rows)

#Run the main loop if this file is executed (self explanatory)
if __name__ == '__main__':
    main_loop()