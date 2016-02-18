#!/bin/bash

{
echo "BLOCK,THREAD,MEM,%TIME,TIME(units...)"
for blk in 32 64 128 256 512 1024; do
  for thd in 32 64 128 256 512 1024; do
    for mem in 0.3 0.7; do
      echo "${blk},${thd},${mem},$(nvprof ./transpose -b ${blk} -t ${thd} -u ${mem} 2>&1 | fgrep transpose_shared | xargs | cut -d' ' -f 1-2 --output-delimiter=',')"
    done
  done
done
} > trans_shared_time.csv
