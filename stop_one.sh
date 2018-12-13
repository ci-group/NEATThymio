#stop python
kill $(ps -u pi | grep python)

#check if asebamedulla is running
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
fi

#stop thymio
(asebamedulla "ser:device=/dev/ttyACM0"&)
python stop_thymio.py