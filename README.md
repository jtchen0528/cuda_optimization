## Reduction
### Single Block
algo|Interleaved|Interleaved_2 | blocked | blocked | half_threads | unroll_last| unroll_last | coalesce
---|---|---|---|---|---|---|---|---
sm|N|N|N|Y|Y|Y|Y|Y
N|1024|1024|1024|1024|2048|2048|2048|8192
T|5.056us|3.169us|2.688us|3.360us|3.392us|3.104us|2.976us|3.072us
Perf|5.056us|3.169us|2.688us|3.360us|1.696us|1.552us|1.488us|0.384us
speedup|1x|1.6x|1.9x|1.5x|3.0x|3.3x|3.4x|13.2x

## Matrix Multiplication
algo|naive|tiled | Transpose | Naive Transposed
---|---|---|---|---
sm|N|Y|Y|N
N|1024 * 1024|1024 * 1024|1024 * 1024|1024 * 1024
T|2.717ms|1.854ms|6.094ms|17.268ms
speedup|1x|1.5x|0.4x|0.2x
<!-- Perf|5.056us|3.169us
speedup|1x|1.6x|1.9x -->

## 2d Convolution

algo|naive|constant memory kernel|tiled shared memory
---|---|---|---
sm|N|N|Y
K| 7 * 7|7 * 7|7*7
N|1024 * 1024 |1024 * 1024|1024 * 1024
T|884,770ns|841,438 ns| 208,321 ns
speedup|1x|1.1x|4.2x
<!-- Perf|5.056us|3.169us
speedup|1x|1.6x|1.9x -->