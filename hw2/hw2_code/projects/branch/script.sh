#!/bin/bash

echo "" > out.txt

for i in `seq 1 64`;
do
	echo "__device__ float bigfunction"$i"(){" >> out.txt
	echo "return (" >> out.txt 
	shuf -n 10 gen.txt | tr '\n' ' ' >> out.txt	
	echo "float(threadIdx.x))))))))))));" >> out.txt
	echo "}" >> out.txt
done
