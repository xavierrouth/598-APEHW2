alias b := build
alias c := clean
alias h := hotspot


hotspot dataset: (perf dataset)
    hotspot --sourcePaths . --appPath . output/perf.data 


perf dataset : (perf_record "./genetic_benchmark " + dataset)
 
time dataset : build
    taskset -c 0,1,2,3,4,5,6,7 ./genetic_benchmark {{dataset}}

perf_record *args: build
    taskset -c 0,1,2,3,4,5,6,7 perf record -o output/perf.data -e cache-misses,cycles,page-faults -F 10000 --call-graph dwarf {{args}}
    
build:
    make -f OriginalMakefile all

clean: clean_src clean_generated

gdb dataset:
    OMP_NUM_THREADS=2 gdb -x ./gdbconfig --args ./genetic_benchmark {{dataset}} 

[working-directory: 'src']
clean_src:
    #!/bin/sh
    make clean

clean_generated:
    rm -f *.exe; rm -f *.obj