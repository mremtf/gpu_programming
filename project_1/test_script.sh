#!/bin/bash

{
for blk in 32 64 128 256 512 1024; do
  for thd in 32 64 128 256 512 1024; do
    for mem in 0.1 0.5 0.9; do
      echo "BLK: ${blk} THD: ${thd} MEM: ${mem}"
      nvprof ./vector_add -b ${blk} -t ${thd} -u ${mem} -v 2>&1 | fgrep cuda_vector_add
    done
  done
done
} > vec_time.txt
