#!/bin/bash

make clean && make
echo "test pi (accuracy to 3.141xxx)"

for i in {1..100};
do
    pi=$(./pi.out 4 100000000 | grep 3.141)
    if [ -z $pi ]; then
        echo "4 thread, pi is not 3.141xxx"
    fi
    pi=$(./pi.out 3 100000000 | grep 3.141)
    if [ -z $pi ]; then
        echo "3 thread, pi is not 3.141xxx"
    fi
done