#!/bin/bash
batch=$1

make ntt_speed
test/test_ntt512 >results/ntt512_${batch}
