import time

class Logger:
    def __init__(self):
        date = time.strftime("%d-%m-%y_%H-%M")
        self.__logger = open("logs/" + date + "_neat", "w")

    def __write__(self, msg):
        self.__logger.write(msg + "\n")

    def debug(self, msg):
        self.__write__("debug: " + msg)

    def critical(self, msg):
        self.__write__("!!CRITICAL!!: " + msg)

    def info(self, msg):
        self.__write__("info: " + msg)

    def warning(self, msg):
        self.__write__("warning!: " + msg)

    def close(self):
        self.__logger.close()

if __name__ == "__main__":
    logger = Logger()
    logger.critical("critical test")
    logger.info("info test")
    logger.debug("debut test")
    logger.__write__("test")
    logger.close()