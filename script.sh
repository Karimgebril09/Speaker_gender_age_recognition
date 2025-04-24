#!/usr/bin/bash

startt=$1
endd=$2
move=$3
delete=$4

for ((i=startt; i<=endd; i++)); do
	if [ "$move" == "true" ]; then
		mv $i/* .. 
	fi

	if [ "$delete" == "true" ]; then
		rm -r $i
	fi	
done
