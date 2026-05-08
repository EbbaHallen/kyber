#!/bin/bash
version=$1

make ntt_speed
test/test_ntt512 >>results/ntt512_${version}
