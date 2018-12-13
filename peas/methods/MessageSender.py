__author__ = 'alessandrozonta'

import threading
import socket
import sys
import traceback
import cPickle as pickle
import struct


def sendOneMessage(conn, data):
    packed_data = pickle.dumps(data)
    length = len(packed_data)
    conn.sendall(struct.pack('!I', length))
    conn.sendall(packed_data)


# Sends outgoing messages to the remote host
class MessageSender(threading.Thread):
    def __init__(self, ipAddress, port, simulationLogger):
        threading.Thread.__init__(self)
        self.__ipAddress = ipAddress
        self.__port = port
        self.__outbox = list()
        self.__outboxNotEmpty = threading.Condition()
        self.__connectionSocket = None
        self.__isStopped = threading.Event()
        self.__simLogger = simulationLogger

    @property
    def ipAddress(self):
        return self.__ipAddress

    def __estabilishConnection(self):
        nAttempt = 0
        if self.__connectionSocket:
            self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - ALREADY CONNECTED')
            return True
        # Otherwise retry to connect unless stop signal is sent
        while not self.__stopped():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.__ipAddress, self.__port))
                self.__connectionSocket = sock
                self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - CONNECTED @ attempt' + str(nAttempt))
                return True
            except:
                # Error during connect, new attempt if not stopped
                nAttempt += 1
        return False

    def outboxAppend(self, item):
        with self.__outboxNotEmpty:
            self.__outbox.append(item)
            #print "Added genotype to the outbox"
            self.__outboxNotEmpty.notify()

    def __outboxPop(self):
        item = None
        with self.__outboxNotEmpty:
            if not self.__outbox:
                self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - EMPTY OUTBOX: WAIT')
                self.__outboxNotEmpty.wait()
            if not self.__stopped():
                self.__simLogger.debug(
                    'Sender - ' + self.__ipAddress + ' - OUTBOX is' + str(self.__outbox) + ' - taking ' + str(
                        self.__outbox[0]))
                item = self.__outbox.pop(0)
        return item

    def run(self):
        try:
            self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - RUNNING')
            while not self.__stopped():
                item = self.__outboxPop()
                self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - OUTBOX popped ' + str(item))
                if item and self.__estabilishConnection():
                    # Not stopped and has an item to send and an estabilished connection
                    try:
                        sendOneMessage(self.__connectionSocket, item)
                        self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - SENT ' + str(item))
                    except:
                        # Error while sending: put back item in the outbox
                        with self.__outboxNotEmpty:
                            self.__outbox.insert(0, item)
                        # Current socket is corrupted: closing it
                        self.__connectionSocket.close()
                        self.__connectionSocket = None
                        self.__simLogger.warning(
                            'Sender - ' + self.__ipAddress +
                            ' - Error while sending - CLOSED socket and restored OUTBOX:' + str(self.__outbox))
            self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - STOPPED -> EXITING...')
        except:
            self.__simLogger.critical(
                'Error in Sender ' + self.__ipAddress + ': ' + str(sys.exc_info()[0]) + ' - ' + traceback.format_exc())

    def stop(self):
        self.__isStopped.set()
        with self.__outboxNotEmpty:
            self.__outboxNotEmpty.notify()

    def __stopped(self):
        return self.__isStopped.isSet()
