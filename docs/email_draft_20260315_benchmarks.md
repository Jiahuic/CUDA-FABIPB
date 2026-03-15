Subject: GPU-FABIPB benchmark update

Hi [Collaborator Name],

I finished a new benchmark sweep for `1a63` using the current GPU-FABIPB code, with mesh density `10`, `10` repeats per point, and tree depths `5-8`.

The main result is that the GPU path is consistently much faster than the CPU serial baseline, and the best practical configuration is the hybrid mode rather than the "all-GPU" mode.

Against CPU serial, the hybrid configuration gives:

- depth 5: `12.23x` total speedup
- depth 6: `11.51x` total speedup
- depth 7: `6.79x` total speedup
- depth 8: `4.05x` total speedup

The GPU acceleration is coming from both major interaction regimes:

- near-field: about `19-26x` speedup for depths `5-8`
- M2L: about `4-5x` speedup for depths `5-8`

The trend is that shallower trees benefit more from the near-field acceleration, while deeper trees shift more of the work into M2L. This supports the current paper direction: the performance gain is coming from the destination-grouped GPU treatment of the dominant interaction stages, while the best overall runtime is obtained with an adaptive hybrid policy rather than forcing every stage onto the GPU.

The benchmark outputs are here:

`/home/jiahuic/Garage/electrostatics/GPU-FABIPB/build/benchmark_matrix/20260315_094824`

The most useful files are:

- `summary.csv`: averaged speedups and stage comparisons
- `results.csv`: averaged per-configuration timings
- `results_raw.csv`: all individual repeats

If you want, I can next turn these results into a compact table for the paper draft.
