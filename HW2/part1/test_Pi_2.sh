#!/bin/bash

cnt=0
echo "Test GO!"
for i in {1..100};
do
    result=$(./pi.out 3 100000000)
    if [ $(grep '3.141' <<< ${result} | wc -l) -eq 1 ]; then
        cnt=$((cnt+1))
    else
        echo "Wrong!"
        echo ${result}
    fi
    echo ${result} >> result.txt
done

echo ${cnt}