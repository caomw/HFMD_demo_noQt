#!/bin/sh
filename=$1
cat ${filename} | while read line
do
  ./createTestList.sh ${line}
done 
