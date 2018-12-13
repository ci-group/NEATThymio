#!/bin/bash

if [ ! -z  $1 ]; then
	BOTS=$1
else
	BOTS=./bots.txt
fi

while read line
do
	echo "SYNC CLOCK AND FILES FOR: $line"
	rsync -rzL --exclude 'output' ./* pi@$line:~/ #copy files only that changed
    #ssh -n pi@$line 'sudo rdate -ncv 0.nl.pool.ntp.org'
	#change the filename to bots.txt
	ssh -n pi@$line 'mv '$BOTS' bots.txt'
    echo "done"
done < $BOTS
wait
