#!/bin/bash

if [ ! -z  $1 ]; then
	BOTS=$1
else
	BOTS=./bots.txt
fi

while read line
do
	echo "STARTING $line"
	scp -r pi@$line:~/output/TEST_LOGGING* .
done < $BOTS
wait
