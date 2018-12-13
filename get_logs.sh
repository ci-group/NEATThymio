#!/bin/bash

if [ ! -z  $1 ]; then
	BOTS=$1
else
	BOTS=./bots.txt
fi

while read line
do
	echo "STARTING $line"
	scp -r pi@$line:~/output/odNEAT_Foraging_ExperimentID4_G_192.168.1.*/odNEAT_Foraging_ExperimentID4_G_192.168.1.* output/thymio/8robots/
done < $BOTS
wait
