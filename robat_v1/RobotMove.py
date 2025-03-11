from thymiodirect import Connection
from thymiodirect import Thymio
import time
import random


class RobotMove():
    def __init__(self, angle_queue, forward_speed, turn_speed, waiturn, left_sensor_threshold, right_sensor_threshold, critical_level, av_above_level, trigger_level, ground_sensors_bool = False):
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        self.counter_turn = waiturn
        self.left_sensor_threshold = left_sensor_threshold
        self.right_sensor_threshold = right_sensor_threshold
        self.ground_sensors_bool = ground_sensors_bool
        self.critical_level = critical_level
        self.trigger_level = trigger_level
        self.av_above_level = av_above_level
        self.angle_queue = angle_queue


        print("Initializing Thymio Robot")
        print('check',self.angle_queue.get())
        port = Connection.serial_default_port()
        th = Thymio(serial_port=port,
                    on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
        # Connect to Robot
        th.connect()
        self.robot = th[th.first_node()]

        # Delay to allow robot initialization of all variables
        time.sleep(1)
        # b) print all variables
        # print(th.variables(th.first_node()))
        
        print("Robot connected")
        print(self.ground_sensors_bool)
        if self.ground_sensors_bool:
            print('ground.delta  L R = ', self.robot['prox.ground.delta'])
            print('ground.reflected  L R = ', self.robot['prox.ground.reflected'])

        self.stop_bool = False

    def audio_move(self):
        #print('self.ange', self.angle)
        # check if the robot is lifted
        if self.check_stop_all_motion():
            self.stop_bool = True
            self.stop()
        self.avoid_white_line()
        if self.angle_queue.get() is None:   #neutral movement
            print('1',self.angle_queue.get())
            self.robot["leds.top"] = [0, 0, 0]
            self.move_forward()
        
        elif self.av_above_level > self.critical_level: #repulsion
            print('2',self.angle_queue.get())
            self.rotate_left()
        else:
            if self.angle_queue.get() < 0:
                print('3',self.angle_queue.get())
                self.rotate_right()
            else:
                print('4',self.angle_queue.get())
                self.rotate_right()

    def move_forward(self):
        if self.robot is not None:
            if self.check_stop_all_motion():
                print("stop all motion: move forward")
                self.stop_bool = True
                # interrupt the loop
                return
            self.robot['motor.left.target'] = self.forward_speed
            self.robot['motor.right.target'] = self.forward_speed
            #if self.obstacle_ahead():
            #    self.collision_avoidance()
            #    continue
        else:
            print("self.robot is None")
            self.stop_bool = True
            self.stop()

    def rotate_right(self):
        if self.robot is not None:
            counter = self.counter_turn
            while counter > 0:
                if self.check_stop_all_motion():
                    
                    self.stop_bool = True
                    break
                print("rotate right")
                self.robot['motor.left.target'] = int(0.5 * 1/90 * (speed*int(self.angle_queue.get())))
                self.robot['motor.right.target'] = -int(0.5 * 1/90 * (speed*int(self.angle_queue.get())))
                # if self.obstacle_ahead():
                #     self.collision_avoidance()
                #     continue
                counter -= 1
            else:
                self.robot['motor.left.target'] = 0
                self.robot['motor.right.target'] = 0

    def rotate_left(self):
        if self.robot is not None:
            counter = self.counter_turn
            while counter > 0:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                if counter == self.counter_turn:
                    print("rotate left")
                self.robot['motor.left.target'] = -int(0.5 * 1/90 * (speed*int(self.angle_queue.get())))
                self.robot['motor.right.target'] = int(0.5 * 1/90 * (speed*int(self.angle_queue.get())))
                # if self.obstacle_ahead():
                #     self.collision_avoidance()
                #     continue
                counter -= 1
            else:
                # print("robot stop")
                self.robot['motor.left.target'] = 0
                self.robot['motor.right.target'] = 0

    def move_back(self): # move back when robot sees a obstacle in front
        if self.robot is not None:
            counter = 4000
            while counter > 0:
                if self.check_stop_all_motion():
                    print("stop all motion: move back")
                    self.stop_bool = True
                    break
                if self.robot['prox.horizontal'][5] > 600 or self.robot['prox.horizontal'][6] > 600:
                    self.stop_bool = True
                    self.stop()
                self.robot['motor.left.target'] = -self.forward_speed
                self.robot['motor.right.target'] = -self.forward_speed
                counter -= 1
            else:
                # print("robot stop")
                self.robot['motor.left.target'] = 0
                self.robot['motor.right.target'] = 0
            self.random_turn()

    def check_stop_all_motion(self):
        if self.robot is not None:
            # print(f"Prox 0: {self.robot['prox.ground.delta'][0]}, Prox 1: {self.robot['prox.ground.delta'][1]}")
            if self.robot['prox.ground.delta'][0] < 100 or self.robot['prox.ground.delta'][1] < 100:
                print("Robot lifted")
                return True
        return False

    def avoid_white_line(self):
        # check if the white line is detected
        if self.robot['prox.ground.reflected'][0] > self.left_sensor_threshold  and self.robot['prox.ground.reflected'][1] > self.right_sensor_threshold:
            # Both sensors detect the line
            print('line detected L and R')
            self.move_back()
            return
        if self.robot['prox.ground.reflected'][0] > self.left_sensor_threshold:
            # Left sensor detects the line
            self.rotate_right()
            print('line detected L')
            counter = self.counter_turn
            while counter > 0:
                if self.check_stop_all_motion():
                    print("stop all motion: rotate right")
                    self.stop_bool = True
                    break
                self.robot['motor.left.target'] = self.turn_speed
                self.robot['motor.right.target'] = -self.turn_speed
                counter -= 1
        if self.robot['prox.ground.reflected'][1] > self.right_sensor_threshold:
            # Right sensor detects the line
            self.rotate_left()      
            print('line detected R')
            counter = self.counter_turn
            while counter > 0:
                if self.check_stop_all_motion():
                    print("stop all motion: rotate right")
                    self.stop_bool = True
                    break
                self.robot['motor.left.target'] = self.turn_speed
                self.robot['motor.right.target'] = -self.turn_speed
                counter -= 1

    def stop(self):
        if self.robot is not None:
            self.robot['motor.left.target'] = 0
            self.robot['motor.right.target'] = 0

#     def collision_avoidance(self):
#         if self.robot is not None:
#             print("Starting Collision avoidance")
#             self.robot['leds.top'] = [32, 0, 0]
#             counter = 3000
#             print("Move back")
#             while counter > 0:
#                 if self.robot['prox.horizontal'][5] > 600 or self.robot['prox.horizontal'][6] > 600:
#                     self.stop()
#                     continue
#                 self.robot['motor.left.target'] = -200
#                 self.robot['motor.right.target'] = -200
#                 counter -= 1
#             counter = 2000
#             print("Turn")
#             while counter > 0:
#                 self.robot['motor.left.target'] = -300
#                 self.robot['motor.right.target'] = 300
#                 counter -= 1
#             self.robot['leds.top'] = [0, 32, 0]

#     def obstacle_ahead(self):
#         if self.robot is not None:
#             if (self.robot['prox.horizontal'][0] > 1000 or self.robot['prox.horizontal'][1] > 1000 or
#                     self.robot['prox.horizontal'][2] > 1000 or self.robot['prox.horizontal'][3] > 1000 or self.robot['prox.horizontal'][4] > 1000):
#                 return True
#         return False

    def random_move(self):
        while True:
            if self.stop_bool:
                print("Stopping Robot")
                self.stop()
                break
            self.move_forward()
            if bool(random.getrandbits(1)):  
                self.rotate_right()
            else:
                self.rotate_left()

    def random_turn(self):
        while True:
            if self.check_stop_all_motion():
                self.stop_bool = True
                break
            if self.stop_bool:
                print("Stopping Robot")
                self.stop()
                break
            if bool(random.getrandbits(1)):  
                print('random turn right')
                self.rotate_right()
                break
            else:
                print('random turn left')
                self.rotate_left()
                break
            