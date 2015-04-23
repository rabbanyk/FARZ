#!/bin/sh
EXP="UndUnw1000S"

mkdir ./synthetic$EXP
mkdir ./synthetic$EXP/data
for j in $(seq 9)
do
	MU=".$j"
	#mkdir ./synthetic$EXP/S$j
	for i in  $(seq 10)
	do 
		./benchmark -N 1000 -k 20  -maxk 50 -t1 2 -t2 1 -mu $MU -minc 20 -maxc 100
		mv ./network.dat ./synthetic$EXP/data/S$j-network$i.dat
		mv ./community.dat ./synthetic$EXP/data/S$j-network$i.gt
	done
done

rm -f statistics.dat
rm -f time_seed.dat



