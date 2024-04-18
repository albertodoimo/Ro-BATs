
from thymiodirect import Connection 
from thymiodirect import Thymio

import argparse
import time
import math

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

        state = "START"
        print(f"State: {state}")

              # Main loop
        while True:  
            if state == "START":
                # Get the value of the rear proximity sensor
                proxl2 = robot['prox.horizontal'][0]
                proxl1 = robot['prox.horizontal'][1]
                proxc = robot['prox.horizontal'][2]
                proxr1 = robot['prox.horizontal'][3]
                proxr2 = robot['prox.horizontal'][4]
                print(proxr2,proxr1,proxc,proxl1,proxl2)
                
                # get lights turn on based on proximity sensors
                if proxl2 > 0:
                    robot["leds.top"] = [0, 0, 255]
                if proxl1> 0:
                    robot["leds.top"] = [0, 255, 255]
                if proxc > 0:
                    robot["leds.top"] = [255, 255, 255]
                if proxr1 > 0:
                    robot["leds.top"] = [255,255,0]
                if proxr2 > 0:
                    robot["leds.top"] = [255, 0, 0]   
                elif proxr2 == 0 and proxr1 == 0 and proxc == 0 and proxl1 == 0 and proxl2 == 0 :
                    robot["leds.top"] = [0, 0, 0]
                     
                     
            
#             # 2.3: indecisive 
#             elif state == "INDECISIVE":
#                 # Get the value of the front proximity sensors
#                 front_sensors = robot["prox.horizontal"][:5]
# 
#                 # Check if any of the front sensors detect an object
#                 if any(sensor_value > 0 for sensor_value in front_sensors):
#                     # If an object is detected, move backward
#                     robot['motor.left.target'] = -200
#                     robot['motor.right.target'] = -200
#                 else:
#                     # If no object is detected, move forward
#                     robot['motor.left.target'] = 200
#                     robot['motor.right.target'] = 200
# 
#             # 2.bonus_1: Shy 
#             elif state == "Shy":
#                 # Get the value of the front proximity sensors
#                 front_sensors = robot["prox.horizontal"][:5]
# 
#                 # Check if any of the front sensors detect an object
#                 if any(sensor_value > 3000 for sensor_value in front_sensors):
#                     # If an object is detected, move backward
#                     robot['motor.left.target'] = 500
#                     robot['motor.right.target'] = -500
#                 elif any(sensor_value > 0 for sensor_value in front_sensors):
#                     # If an object is detected, move backward
#                     robot['motor.left.target'] = 200
#                     robot['motor.right.target'] = -200
#                 else:
#                     # If no object is detected, move forward
#                     robot['motor.left.target'] = 200
#                     robot['motor.right.target'] = 200
# 
#             # 2.bonus_2: Curious 
#             elif state == "Curious":
#                 # Get the value of the front proximity sensors
#                 front_4_sensors = robot["prox.horizontal"][:4]
#                 front_5_sensor = robot["prox.horizontal"][4]
#                 front_all_sensors = robot["prox.horizontal"][:5]
#                 # Check if any of the front sensors detect an object
#                 if any(sensor_value > 0 for sensor_value in front_4_sensors):
#                     if (front_5_sensor >1500):
#                     # If an object is detected, move backward
#                         robot['motor.left.target'] = -200
#                         robot['motor.right.target'] = 200
#                     elif (front_5_sensor >500):
#                         robot['motor.left.target'] = 200
#                         robot['motor.right.target'] = 200
#                     else:
#                         robot['motor.left.target'] = 200
#                         robot['motor.right.target'] = -200
# 
#                 else:
#                     # If no object is detected, move forward
#                     robot['motor.left.target'] = 200
#                     robot['motor.right.target'] = 200
                  
    except Exception as err:
        # Stop robot
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0 
        print(err)


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