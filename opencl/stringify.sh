#!/bin/bash

IN=$1
NAME=${IN%.cl}
OUT=$NAME.h

echo "const char * source =" >$OUT
sed -e 's/\r$//' \
    -e '/^[[:space:]]*$/d' \
    -e 's/\\/\\\\/g' \
    -e 's/"/\\"/g' \
    -e 's/^[[:space:]]*/    "/' \
    -e 's/$/"/' \
    "$IN" >> "$OUT"

echo ";" >>$OUT