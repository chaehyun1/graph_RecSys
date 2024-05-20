#!/bin/sh
# 수정하기 
for i in 0.0 0.5 1.0 1.5 2.0 2.5 3.0 4.0 
do
   --python main.py --dataset=CAMRa2011 --verbose=0 --group_alpha=$i --alpha=0.6 --power=0.7
done