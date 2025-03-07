from thymiodirect import Connection
from thymiodirect import Thymio
import time
import random


class RandomRobotMove():
    def __init__(self):
        print("Initializing Thymio Robot")
        port = Connection.serial_default_port()
        th = Thymio(serial_port=port,
                    on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
        # Connect to Robot
        th.connect()
        self.robot = th[th.first_node()]

        # Delay to allow robot initialization of all variables
        time.sleep(1)
        # b) print all variables
        print(th.variables(th.first_node()))
        print("Robot connected")
        self.stop_bool = False

    def move_forward(self):
        if self.robot is not None:
            counter = 20000
            while counter > 0:
                if self.check_stop_all_motion():
                    print("stop all motion: move forward")
                    self.stop_bool = True
                    break
                self.robot['motor.left.target'] = 200
                self.robot['motor.right.target'] = 200
                if self.obstacle_ahead():
                    self.collision_avoidance()
                    continue
                counter -= 1

            else:
                self.robot['motor.left.target'] = 0
                self.robot['motor.right.target'] = 0

    def rotate_right(self):
        if self.robot is not None:
            counter = 5000
            while counter > 0:
                if self.check_stop_all_motion():
                    print("stop all motion: rotate right")
                    self.stop_bool = True
                    break
                self.robot['motor.left.target'] = 200
                self.robot['motor.right.target'] = -200
                if self.obstacle_ahead():
                    self.collision_avoidance()
                    continue
                counter -= 1
            else:
                self.robot['motor.left.target'] = 0
                self.robot['motor.right.target'] = 0

    def rotate_left(self):
        if self.robot is not None:
            counter = 5000
            while counter > 0:
                if self.check_stop_all_motion():
                    print("stop all motion: rotate left")
                    self.stop_bool = True
                    break
                self.robot['motor.left.target'] = -200
                self.robot['motor.right.target'] = 200
                if self.obstacle_ahead():
                    self.collision_avoidance()
                    continue
                counter -= 1
            else:
                # print("robot stop")
                self.robot['motor.left.target'] = 0
                self.robot['motor.right.target'] = 0

    def move_back(self):
        if self.robot is not None:
            counter = 20000
            while counter > 0:
                if self.check_stop_all_motion():
                    print("stop all motion: move back")
                    self.stop_bool = True
                    break
                if self.robot['prox.horizontal'][5] > 600 or self.robot['prox.horizontal'][6] > 600:
                    self.stop()
                    continue
                self.robot['motor.left.target'] = -200
                self.robot['motor.right.target'] = -200
                counter -= 1
            else:
                # print("robot stop")
                self.robot['motor.left.target'] = 0
                self.robot['motor.right.target'] = 0

    def check_stop_all_motion(self):
        if self.robot is not None:
            # print(f"Prox 0: {self.robot['prox.ground.delta'][0]}, Prox 1: {self.robot['prox.ground.delta'][1]}")
            if self.robot['prox.ground.delta'][0] < 500 or self.robot['prox.ground.delta'][1] < 500:
                print("Robot lifted")
                return True
        return False

    def stop(self):
        if self.robot is not None:
            self.robot['motor.left.target'] = 0
            self.robot['motor.right.target'] = 0

    def collision_avoidance(self):
        if self.robot is not None:
            print("Starting Collision avoidance")
            self.robot['leds.top'] = [32, 0, 0]
            counter = 3000
            print("Move back")
            while counter > 0:
                if self.robot['prox.horizontal'][5] > 600 or self.robot['prox.horizontal'][6] > 600:
                    self.stop()
                    continue
                self.robot['motor.left.target'] = -200
                self.robot['motor.right.target'] = -200
                counter -= 1
            counter = 2000
            print("Turn")
            while counter > 0:
                self.robot['motor.left.target'] = -300
                self.robot['motor.right.target'] = 300
                counter -= 1
            self.robot['leds.top'] = [0, 32, 0]

    def obstacle_ahead(self):
        if self.robot is not None:
            if (self.robot['prox.horizontal'][0] > 1000 or self.robot['prox.horizontal'][1] > 1000 or
                    self.robot['prox.horizontal'][2] > 1000 or self.robot['prox.horizontal'][3] > 1000 or self.robot['prox.horizontal'][4] > 1000):
                return True
        return False

    def random_move(self):
        while True:
            if self.stop_bool:
                print("Stopping Robot")
                self.stop()
                break
            self.move_forward()
            if self.stop_bool:
                print("Stopping Robot")
                self.stop()
                break
            if bool(random.getrandbits(1)):
                self.rotate_right()
            else:
                self.rotate_left()
        self.stop_bool = False
