## Profiling
This is for profiling each of the different kernels.

#### TODO:
1. Incorporate warmup logic.
2. Parameter sweeping.


Launch all test cases with nsys. Traces will be stored in this directory.
```
./launch_all_trace.sh ../../build/
```

Checking memory
```
compute-sanitizer --tool memcheck <KERNEL>
compute-sanitizer --tool racecheck <KERNEL>
compute-sanitizer --tool initcheck <KERNEL>
```

Systems
```
nsys profile -o run --trace=cuda,nvtx <KERNEL>
```

Compute
```
ncu --set=quick --section=SpeedOfLight <KERNEL>
ncu --kernel-name "KERNEL" --set=full --replay-mode=kernel <KERNEL>
```

PTX/SASS
```
nvdisasm --print-line-info kernel.cubin | less
cuobjdump --dup-sass KERNEL | less
```
