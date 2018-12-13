import cv2
import numpy as np
from cameracontroller import CameraController
import time
from helpers import *
import logging

class ObjectDetector():

    def __init__(self, color_threshold, amount_threshold, thymioController):
        self.color_threshold = color_threshold
        self.TIMER = 10
        max_motor_speed= 200

        self.red_color_threshold = None
        self.amount_threshold = amount_threshold
        self.camera = CameraController()
        self.thymioController = thymioController

        self.checkWheels()
        self.has_goal_threshold = None
        
        self.has_puck_threshold = self.calibrate_has_puck2()
        self.has_goal_threshold2 = self.calibrate_has_goal3()*1.0
        #self.has_puck_threshold =   0.167 ## WHAT ARE ACTUAL VALUES??
        #self.has_goal_threshold2 =  0.445 ## WHAT ARE ACTUAL VALUES??

        self.checkCalibration()
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('simulationLogger')

        logger.info('puck threshold is')
        logger.info(str(self.has_puck_threshold))
        logger.info('goal threshold is')
        logger.info(str(self.has_goal_threshold2))


    def checkWheels(self):
        left = 1.0
        right = 1.0
        motorspeed = { 'left': left, 'right': right }
        writeMotorSpeed(self.thymioController, motorspeed, max_motor_speed)
        left = -1.0
        right = -1.0
        time.sleep(5)
        motorspeed = { 'left': left, 'right': right }
        writeMotorSpeed(self.thymioController, motorspeed, max_motor_speed)
        time.sleep(5)
        stopThymio(self.thymioController)
        time.sleep(self.TIMER)

    def checkCalibration(self):
        puck_center, puck_size = self.detect_puck()
        goal_center, goal_size = self.detect_goal()
        counter = 0
        hasPuck=False
        seeGoal = False
        while not (seeGoal and hasPuck and counter > 3):
            puck_center, puck_size = self.detect_puck()
            goal_center, goal_size = self.detect_goal()

            if (seeGoal and hasPuck and (sum(self.thymioController.GetVariable("thymio-II", "prox.ground.reflected")) < 1000) and (min(self.thymioController.GetVariable("thymio-II", "prox.ground.reflected")) < 380)):
                counter += 1
            else:
                counter = 0

            if (hasPuck and goal_size > 0.01):
                seeGoal = True
            else:
                seeGoal = False

            if (puck_center[0] < self.has_puck_threshold and puck_size > 0.05):
                hasPuck = True
                #else:
                #hasPuck = False

            if(puck_size > 0.01 or hasPuck):
                seePuck = True
            else:
                seePuck = False

            if seePuck:
                if hasPuck:
                    if seeGoal:
                        motorspeed = { 'left': 1.0, 'right': 1.0 }
                    
                    else:
                        motorspeed = { 'left': 0.4, 'right': -0.4 }
                else:
                    motorspeed = { 'left': 1.0, 'right': 1.0 }
            else:
                motorspeed = { 'left': 0.4, 'right': -0.4 }

            writeMotorSpeed(self.thymioController, motorspeed, max_motor_speed)
        
            if seePuck and not hasPuck and not seeGoal:
                self.thymioController.SendEventName('SetColor', [0, 32, 0, 0], reply_handler=dbusReply, error_handler=dbusError)
            if hasPuck and not seeGoal:
                self.thymioController.SendEventName('SetColor', [0, 0, 32, 0], reply_handler=dbusReply, error_handler=dbusError)
            if seeGoal:
                self.thymioController.SendEventName('SetColor', [32, 0, 0, 0], reply_handler=dbusReply, error_handler=dbusError)

        self.thymioController.SendEventName('PlayFreq', [700, 0])
        time.sleep(1)
        self.thymioController.SendEventName('PlayFreq', [0, -1])
        stopThymio(self.thymioController)
        time.sleep(20)



    def calibrate_has_puck(self):
            raw_input("Put the puck in front of the camera and press enter.")
            puck_center1, _ = self.detect_puck()
            raw_input("Put the puck just out of reach and press enter.")
            puck_center2, _ = self.detect_puck()
            raw_input("Remove the puck and press enter.")
            if puck_center1[0] < puck_center2[0]:
                return (puck_center1[0] + puck_center2[0]) / 2.0
            else:
                return puck_center2[0]

    def calibrate_has_puck2(self):
            self.thymioController.SendEventName('PlayFreq', [700, 0])
            puck_center1, _ = self.detect_puck()
            time.sleep(1)
            self.thymioController.SendEventName('PlayFreq', [0, -1])
            time.sleep(self.TIMER)
            self.thymioController.SendEventName('PlayFreq', [700, 0])
            puck_center2, _ = self.detect_puck()
            time.sleep(1)
            self.thymioController.SendEventName('PlayFreq', [0, -1])
            time.sleep(self.TIMER)
            if puck_center1[0] < puck_center2[0]:
                return (puck_center1[0] + puck_center2[0]) / 2.0
            else:
                return puck_center2[0]


    def calibrate_has_goal(self):
        raw_input("Put the puck in front of the camera and the goal and press enter.")
        goal_center1, _ = self.detect_goal()
        raw_input("Put the puck in front of the camera and just out of reach  of the goal and press enter.")
        goal_center2, _ = self.detect_goal()
        raw_input("Remove the puck and press enter.")
        if goal_center1[0] < goal_center2[0]:
            return (goal_center1[0] + goal_center2[0]) / 2.0
        else:
            return goal_center2[0]


    def calibrate_has_goal3(self):
        self.thymioController.SendEventName('PlayFreq', [700, 0])
        time.sleep(1)
        self.thymioController.SendEventName('PlayFreq', [0, -1])

        img = self.camera.getCurrentImage()
        img = img.astype('float32')
        img1 = img[:, :, 2] / np.sum(img, 2)
        median_red = np.mean(img1[-20:, :])

        self.red_color_threshold = median_red*0.9

        goal_size_list = []
        wait_time = time.time()
        while time.time() - wait_time < 20:
            goal_center1, goal_size1 = self.detect_goal()
            if goal_size1 > -1.0:
                goal_size_list.append(goal_size1)
        self.thymioController.SendEventName('PlayFreq', [700, 0])
        time.sleep(self.TIMER)
        self.thymioController.SendEventName('PlayFreq', [0, -1])
        wait_time = time.time()
        while time.time() - wait_time < 20:
            goal_center2, goal_size2 = self.detect_goal()
            if goal_size2 > -1.0:
                goal_size_list.append(goal_size2)
        self.thymioController.SendEventName('PlayFreq', [700, 0])
        time.sleep(1)
        self.thymioController.SendEventName('PlayFreq', [0, -1])
        time.sleep(self.TIMER)
        return sum(goal_size_list) / float(len(goal_size_list))



    def detect_puck(self):
        img = self.camera.getCurrentImage()
        img = img.astype('float32')
        img1 = img[:, :, 1] / np.sum(img, 2)
        vals = np.where(img1 > self.color_threshold)
        middle = (np.mean(vals[0]) / img1.shape[0], np.mean(vals[1]) / img1.shape[1])
        percentage = float(len(vals[0])) / float(img1.shape[0] * img1.shape[1])
        if percentage < self.amount_threshold:
            middle = (-1,-1)
            percentage = -1
        return middle, percentage


    def detect_goal(self):
        img = self.camera.getCurrentImage()
        img = img.astype('float32')
        img1 = img[:, :, 2] / np.sum(img, 2)
        vals = np.where(img1 > self.red_color_threshold)
        middle = (np.mean(vals[0]) / img1.shape[0], np.mean(vals[1]) / img1.shape[1])
        percentage = float(len(vals[0])) / float(img1.shape[0] * img1.shape[1])
        if percentage < self.amount_threshold:
            middle = (-1,-1)
            percentage = -1
        return middle, percentage

    def draw_rectangle(self, image, middle, percentage):
        if len(middle) == 2:
            y_pixels = int(np.sqrt(percentage)* image.shape[0])
            x_pixels = int(np.sqrt(percentage)* image.shape[1])
            point1 = (int(middle[1]-0.5*x_pixels), int(middle[0]-0.5*y_pixels))
            point2 = (int(middle[1]+0.5*x_pixels), int(middle[0]+0.5*y_pixels))
            cv2.rectangle(image, point1, point2, (1,1,255))
        return image

    def BGRtobgr(self, image):
        # Get full intensity picture.
        image = image.astype('float32')
        full_intensity = np.sum(image, 2)
        image /= np.dstack((full_intensity, full_intensity, full_intensity))
        image *= 255.0
        image = image.astype('uint8')
        return image


if __name__ == '__main__':
    import os
    x = 0
    folder = "/home/bart/Documents/fotos_zonder_licht/"
    for file in os.listdir(folder):
        x += 1
        image = cv2.imread(os.path.join(folder,file))
        _, middle, percentage = detect_puck(image, 0.40, 0.0001)
        _, middle2, percentage2 = detect_goal(image, 0.4, 0.05)
        image = draw_rectangle(image, middle, percentage)
        image = draw_rectangle(image, middle2, percentage2)
        cv2.imwrite('/home/bart/Documents/objects/object_detection' + str(x) + '.jpg', image)
