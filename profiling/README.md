## Profiling
This is for profiling each of the different kernels.

#### TODO:
1. Incorporate warmup logic.
2. Maybe write a few helper bash scripts for this.
3. Parameter sweeping.

Just a few notes.

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