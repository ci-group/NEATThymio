#!/bin/bash

killall -s9 asebamedulla
sleep 2

if ps aux | grep "[a]sebamedulla" > /dev/null
then
    echo "asebamedulla is already running"
else
	for port in 1337 31337
	do
		process=`fuser -n tcp -k $port | sed "s/$port\/tcp:[[:space:]]*//"`
		if [ -n "$process" ]; then
			kill -9 $process
		fi
	done
    eval $(dbus-launch --sh-syntax)
    export DBUS_SESSION_BUS_ADDRESS
    export DBUS_SESSION_BUS_PID
    (asebamedulla "ser:device=/dev/ttyACM0" &)

    #above all for killing processes that are maybe still open
    python $@
fi
