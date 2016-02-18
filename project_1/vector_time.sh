#!/bin/bash

{
echo "BLOCK,THREAD,MEM,%TIME,TIME(units...)"
for blk in 32 64 128 256 512 1024; do
  for thd in 32 64 128 256 512 1024; do
    for mem in 0.1 0.3 0.5 0.7 0.9; do
      echo "${blk},${thd},${mem},$(nvprof ./vector_add -b ${blk} -t ${thd} -u ${mem} -v 2>&1 | fgrep cuda_vector_add | xargs | cut -d' ' -f 1-2 --output-delimiter=',')"
    done
  done
done
} > vec_time.csv
