# imports
import argparse
import time
import math
import random

# import thymiodirect package
from thymiodirect import Connection 
from thymiodirect import Thymio

def main(use_sim=False, ip='localhost', port=2001):
    ''' Main function '''

    try:
        # Configure Interface to Thymio robot
        # simulation
        if use_sim:
            th = Thymio(use_tcp=True, host=ip, tcp_port=port, 
                        on_connect=lambda node_id: print(f' Thymio {node_id} is connected'))
        # real robot
        else:
            port = Connection.serial_default_port()
            th = Thymio(serial_port=port, 
                        on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))

        # Connect to Robot
        th.connect()
        robot = th[th.first_node()]

        # Delay to allow robot initialization of all variables
        time.sleep(1)

        # additional constants
        straightSpeed = 100
        turnSpeed = 80
        
        #TODO: 4.1 variables

        state = 'find'
        ts = time.time()
        direction = random.choice(['left', 'right'])
        
        # Main loop
        while True:

            # get input
            detectsCollision = max([robot['prox.horizontal'][i] > 1000 for i in range(5)]) # check if there is a near obstacle
            detectsLine = [robot['prox.ground.reflected'][i] <= 500 for i in range(2)] # get line check inputs

            # update state
            if detectsCollision: # check for obstacle
                state = 'avoid'
            elif not max(detectsLine): # see no line is detected 
                # update start timer and random direction if previous state wasn't find
                if state != 'find':
                    direction = random.choice(['left', 'right'])
                    ts = time.time()
                state = 'find'
            else: # line detected and no collision, hence follow line
                state = 'follow'
            

            #TODO: 4.2 collision avoidance
            if state == 'avoid':
                # rotate right
                robot['motor.left.target'] = turnSpeed
                robot['motor.right.target'] = -turnSpeed

                continue
        

            #TODO: 4.3 random walk
            if state == 'find':

                curTime = time.time()
                if (curTime - ts) % 6.0 < 5.0: # check phase
                    # forward
                    robot['motor.left.target'] = straightSpeed
                    robot['motor.right.target'] = straightSpeed
                else:
                    # turning random direction
                    right = direction == 'right'
                    robot['motor.left.target'] = (-1 + 2 * right) * turnSpeed * 2
                    robot['motor.right.target'] = (1 - 2 * right) * turnSpeed * 2
                
                continue

            #TODO: 4.4 line following

            leftLine = detectsLine[0]
            rightLine = detectsLine[1]

            if leftLine and rightLine:
                # forward
                robot['motor.left.target'] = straightSpeed
                robot['motor.right.target'] = straightSpeed
            else:
                # rotate accordingly
                robot['motor.left.target'] = (-1 + 2 * rightLine) * turnSpeed
                robot['motor.right.target'] = (1 - 2 * rightLine) * turnSpeed
                        
    except Exception as err:
        # Stop robot
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0 
        print(err)
    except KeyboardInterrupt:
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0
        print("Press Ctrl-C again to end the program")


if __name__ == '__main__':
    # Parse commandline arguments to cofigure the interface for a simulation (default = real robot)
    parser = argparse.ArgumentParser(description='Configure optional arguments to run the code with simulated Thymio. '
                                                    'If no arguments are given, the code will run with a real Thymio.')
    
    # Add optional arguments
    parser.add_argument('-s', '--sim', action='store_true', help='set this flag to use simulation')
    parser.add_argument('-i', '--ip', help='set the TCP host ip for simulation. default=localhost', default='localhost')
    parser.add_argument('-p', '--port', type=int, help='set the TCP port for simulation. default=2001', default=2001)

    # Parse arguments and pass them to main function
    args = parser.parse_args()
    main(args.sim, args.ip, args.port)
