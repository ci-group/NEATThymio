# -*- coding: utf-8 -*-
import picamera
import cv2
import numpy as np
import sys, time, io, os
import dbus
import dbus.mainloop.glib
from helpers import *


class CameraController():

    def __init__(self):
        # connect with the Thymio
        CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
        AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()
        self.thymioController = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')
        self.thymioController.LoadScripts(AESL_PATH, reply_handler=dbusReply, error_handler=dbusError)

        # switch thymio LEDs off
        self.thymioController.SendEventName('SetColor', [0, 0, 0, 0], reply_handler=dbusReply, error_handler=dbusError)

        # start the camera
        self.camera = picamera.PiCamera()
        self.camera.resolution = (320, 240)
        self.camera.framerate = 80
        self.camera.start_preview()
        time.sleep(2)
        self.counter = 0
        self.stream = io.BytesIO()


    def getCurrentImage(self):
        self.stream = io.BytesIO()
        for foo in self.camera.capture_continuous(self.stream, format='jpeg', use_video_port=True):
            data = np.fromstring(self.stream.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(data, 1)
            break

        self.counter += 1
        #cv2.imwrite('camera/kut' + str(self.counter) + '.jpg', image)
        return image;
