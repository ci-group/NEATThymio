import sys, os, errno, math, random
import numpy as np

max_motor_speed=200
RAND_MAX = sys.maxint
LEFT = 0
RIGHT = 1

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# Uniform distribution (0..1]
def drand():
    return random.randint(0, RAND_MAX) / float(RAND_MAX + 1)

# Normal distribution, centered on 0, std dev 1
def random_normal():
    return -2 * math.log(drand())

# Used because else webots gives a strange segfault during cross compilation
def sqrt_rand_normal():
    return math.sqrt(random_normal())

def gaussrand():
    return sqrt_rand_normal() * math.cos(2 * math.pi * drand())

def getNextIDPath(path):
    nextID = 0
    filelist = sorted(os.listdir(path))
    if filelist and filelist[-1][0].isdigit():
        nextID = int(filelist[-1][0]) + 1
    return str(nextID)

def writeMotorSpeed(controller, motorspeed, max_speed=max_motor_speed):
    controller.SetVariable("thymio-II", "motor.left.target", [motorspeed['left'] * max_speed])
    controller.SetVariable("thymio-II", "motor.right.target", [motorspeed['right'] * max_speed])


def getProxReadings(controller, ok_callback, nok_callback):
    controller.GetVariable("thymio-II", "prox.horizontal", reply_handler=ok_callback, error_handler=nok_callback)

def getProxReadings(controller, ok_callback, nok_callback):
    controller.GetVariable("thymio-II", "prox.horizontal", reply_handler=ok_callback, error_handler=nok_callback)

def getGroundReadings(controller, ok_callback, nok_callback):
    controller.GetVariable("thymio-II", "prox.ground.reflected", reply_handler=ok_callback, error_handler=nok_callback)

def stopThymio(controller):
    writeMotorSpeed(controller, { 'left': 0, 'right': 0 })

def dbusReply():
    pass


def dbusError(e):
    print 'error %s' % str(e)

def check_stop(task):
    global ctrl_client
    f = ctrl_client.makefile()
    line = f.readline()
    if line.startswith('stop'):
        release_resources(task.thymioController)
        task.exit(0)
    task.ctrl_thread_started = False


def send_image(client, binary_channels, energy, box_dist, goal_dist, boundary='thymio'):
    red = np.zeros(binary_channels[0].shape, np.uint8)
    cv2.putText(red, 'E: {0:.2f} P: {1:.0f} G: {2:.0f}'.format(energy, box_dist, goal_dist), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, ), 1, 255)
    image = np.dstack(binary_channels + [red])
    _, encoded = cv2.imencode('.png', image)
    image_bytes = bytearray(np.asarray(encoded))
    client.send("Content-type: image/png\r\n")
    client.send("Content-Length: %d\r\n\r\n" % len(image_bytes))
    client.send(image_bytes)
    client.send("\r\n--" + boundary + "\r\n")

def release_resources(thymio, ctrl_serversocket, ctrl_client, img_serversocket=None, img_client=None):
    ctrl_serversocket.close()
    if ctrl_client: ctrl_client.close()

    if img_serversocket: img_serversocket.close()
    if img_client: img_client.close()

    stopThymio(thymio)

def write_header(client, boundary='thymio'):
    client.send("HTTP/1.0 200 OK\r\n" +
        "Connection: close\r\n" +
        "Max-Age: 0\r\n" +
        "Expires: 0\r\n" +
        "Cache-Control: no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0\r\n" +
        "Pragma: no-cache\r\n" +
        "Content-Type: multipart/x-mixed-replace; " +
        "boundary=" + boundary + "\r\n" +
        "\r\n" +
        "--" + boundary + "\r\n")