#!/bin/bash

BOTS=./bots.txt

#sync the clocks
sh ./sync_all.sh $BOTS;

while read line
do

	if [ ! -f "distances.p" ]; then # only if not existing
		python dist_angle_matrices.py
	fi

	# current commit git
#	git_sha="$(git rev-parse --short HEAD)"
	git_sha="GO"

	printf "RUNNING: ssh -X pi@$line sh ./start_one.sh $1 $line $git_sha ${2:-foo} ${3:-foo} \n"
	ssh -X pi@$line sh './start_one.sh' $1 $line $git_sha ${2:-foo} ${3:-foo} &  # first command is py file with task
done < $BOTS
wait
