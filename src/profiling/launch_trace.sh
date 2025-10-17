#!/bin/bash
BIN_LOCATION=$1
OUT_FILE=$(basename $BIN_LOCATION)
nsys profile -o $OUT_FILE --force-overwrite=true --trace=cuda,nvtx $BIN_LOCATION