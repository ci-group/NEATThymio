#!/bin/bash

while read line
do
	printf "RUNNING: ssh -X pi@$line sh stop_one.sh\n"
	ssh -X pi@$line sh stop_one.sh &
done < ./bots.txt
wait
