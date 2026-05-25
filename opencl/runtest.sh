#!/bin/bash
version=$1

make ntt_speed
test/test_ntt512.exe >>results/ntt512_${version}
