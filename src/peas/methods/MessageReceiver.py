__author__ = 'alessandrozonta'

import threading
import select
import sys
import traceback
import pickle
import struct


# Waits for incoming messages on one socket and stores them in the shared inbox
class MessageReceiver(threading.Thread):
    def __init__(self, ipAddress, inbox, simulationLogger):
        threading.Thread.__init__(self)
        self.__ipAddress = ipAddress
        self.__inbox = inbox
        self.__connectionSocket = None
        self.__stopSocket = None
        self.__isSocketAlive = threading.Condition()
        self.__isStopped = threading.Event()
        self.__simLogger = simulationLogger

    def get_ip(self):
        return self.__ipAddress

    @staticmethod
    def recvall(conn, count):
        buf = b''
        while count:
            newbuf = conn.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf


    @staticmethod
    def recvOneMessage(socket):
        lengthbuf = MessageReceiver.recvall(socket, 4)
        length, = struct.unpack('!I', lengthbuf)
        data = pickle.loads(MessageReceiver.recvall(socket, length))
        return data

    @property
    def ipAddress(self):
        return self.__ipAddress

    def setConnectionSocket(self, newSock):
        with self.__isSocketAlive:
            if not self.__connectionSocket and newSock:
                self.__connectionSocket = newSock
                self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - CONNECTED!!!')
                self.__isSocketAlive.notify()

    def setStopSocket(self, stopSock):
        self.__stopSocket = stopSock

    def run(self):
        try:
            self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - RUNNING')

            while not self.__stopped():

                # Waits while the connection is not set
                with self.__isSocketAlive:
                    if not self.__connectionSocket and not self.__stopped():
                        self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - Not connected: WAIT')
                        self.__isSocketAlive.wait()

                if not self.__stopped():
                    self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - Waiting on select')
                    readable, _, _ = select.select([self.__connectionSocket], [], [])
                    if self.__stopSocket in readable:
                        # 	Received a message (stop) from localhost
                        self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - StopSocket is in readable')
                        data = MessageReceiver.recvOneMessage(self.__stopSocket)
                        self.__simLogger.debug('Received ' + data)
                    elif self.__connectionSocket in readable:
                        self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - ConnectionSocket is in readable')
                        # 	Received a message from remote host
                        try:
                            data = MessageReceiver.recvOneMessage(self.__connectionSocket)
                            self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - Received ' + str(data))
                            if data and not self.__stopped():
                                # self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - Appending ' + str(data))
                                self.__inbox.append(data)
                                self.__simLogger.debug(
                                    'Receiver - ' + self.__ipAddress + ' - Appended ' + str(data) + ' to inbox.')
                        except:
                            # Error while receiving: current socket is corrupted -> closing it
                            self.__simLogger.warning(
                                'Receiver - ' + self.__ipAddress + ' - Error while receiving - CLOSING socket!' + str(
                                    sys.exc_info()[0]) + ' - ' + traceback.format_exc())
                            self.__connectionSocket.close()
                            self.__connectionSocket = None
        except:
            self.__simLogger.critical('Error in Receiver ' + self.__ipAddress + ': ' + str(
                sys.exc_info()[0]) + ' - ' + traceback.format_exc())

        self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - STOPPED -> EXITING...')

    def stop(self):
        self.__isStopped.set()
        with self.__isSocketAlive:
            self.__isSocketAlive.notify()

    def __stopped(self):
        return self.__isStopped.isSet()