## Reduction
### Single Block
algo|Interleaved|Interleaved_2 | blocked | blocked | half_threads | unroll_last| unroll_last | coalesce
---|---|---|---|---|---|---|---|---
sm|N|N|N|Y|Y|Y|Y|Y
GridDim|1|1|1|1|1|1|1|1
N|1024|1024|1024|1024|2048|2048|2048|8192
T|5.056us|3.169us|2.688us|3.360us|3.392us|3.104us|2.976us|3.072us
Perf|5.056us|3.169us|2.688us|3.360us|1.696us|1.552us|1.488us|0.384us
speedup|1x|1.6x|1.9x|1.5x|3.0x|3.3x|3.4x|13.2x

## Matrix Multiplication
algo|naive|tiled 
---|---|---
sm|N|Y|Y
GridDim|1024|1024|1024
N|1024 * 1024|1024 * 1024
T|2.717ms|1.854ms
speedup|1x|1.5x
<!-- Perf|5.056us|3.169us
speedup|1x|1.6x|1.9x -->

## 2d Convolution

algo|naive|constant memory kernel|tiled shared memory
---|---|---|---
sm|N|N|Y
K| 7 * 7|7 * 7|7*7
GridDim|1024|1024|1024
N|1024 * 1024 |1024 * 1024|1024 * 1024
T|884,770ns|841,438 ns| 208,321 ns
speedup|1x|1.1x|4.2x
<!-- Perf|5.056us|3.169us
speedup|1x|1.6x|1.9x -->

## Exclusive Scan

algo|naive|naive sm|warp
---|---|---|---
sm|N|Y|Y
GridDim|1|1|1
N| 1024 | 1024 | 1024
T|9,024 ns|4,992ns|2,752 ns
speedup|1x|1.81x|3.3x

## Sort

algo|naive|naive warp diverge optimized|naive warp sm|naive warp sm coleased|naive bitonic sort|naive bitonic sort sm|naive merge sort
---|---|---|---|---|---|---|---
sm|N|N|Y|Y|N|Y|N
N|1024|1024|1024|1024|1024|1024|1024
T|253,024ns|157,216ns|90,336ns|90,752ns|35,648ns|31,552ns|142,720ns+11,872ns

* note: coleased use half threads but sync waits. No hiding in swapping 2 elements
* bitonic sort can be optimized with warp divergence

## copy
algo|naive|naive colease
---|---|---
N|1024*1024|1024*1024
T|27,424ns|21,920ns

## transpose
algo|transpose Naive|transpose coleased|transposed coleased no bank conflict
---|---|---|---
N|1024*1024|1024*1024|1024*1024
T|71,264ns|26,720ns|21,088ns
