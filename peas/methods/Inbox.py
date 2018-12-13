__author__ = 'alessandrozonta'

import threading


# Represents a shared inbox object
class Inbox(object):
    def __init__(self, simulationLogger):
        self.__inbox = list()
        self.__inboxLock = threading.Lock()
        self.__simLogger = simulationLogger

    def append(self, data):
        with self.__inboxLock:
            self.__inbox.append(data)

    def popAll(self):
        itemsList = list()
        with self.__inboxLock:
            for i in self.__inbox:
                item = self.__inbox.pop(0)
                # self.__simLogger.debug("popAll - message fitness = " + str(item.fitness))
                itemsList.append(item)
        # self.__simLogger.debug("popAll - Popped " + str(itemsList))
        return itemsList
