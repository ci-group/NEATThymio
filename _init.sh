#!/bin/sh
sleep 2
if ps aux | grep "[a]sebamedulla" > /dev/null
then
    echo "asebamedulla is already running"
else
    eval $(dbus-launch --sh-syntax)
    export DBUS_SESSION_BUS_ADDRESS
    export DBUS_SESSION_BUS_PID
    (asebamedulla "ser:device=/dev/ttyACM0" &)
#    python algorithmForaging.py
fi