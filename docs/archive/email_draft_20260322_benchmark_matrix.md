Subject: Updated benchmark matrix on 1a63 with new GPU setup path

Hi [Collaborator Name],

I finished a new benchmark matrix on `1a63` using the current implementation after the recent setup-side GPU work. The main point is that the new default GPU path now includes the disjoint `panelIA0()` acceleration in both near-field setup and preconditioner setup when `qOrder=1`.

For the depth sweep (`nLev = 5, 6, 7, 8`), the main comparison is:

- `depth=5`: CPU serial `180.36 s`, GPU full `8.68 s`, hybrid best `5.61 s`
- `depth=6`: CPU serial `56.35 s`, GPU full `3.27 s`, hybrid best `3.09 s`
- `depth=7`: CPU serial `33.78 s`, GPU full `5.06 s`, hybrid best `5.92 s`
- `depth=8`: CPU serial `42.95 s`, GPU full `10.97 s`, hybrid best `11.97 s`

So the best overall regime for this case is still around `depth=6`, where the hybrid mode is fastest. At that depth, the speedup is about `18.2x` in total runtime relative to CPU serial. The stage-level behavior is consistent with what we saw before: the main GPU gains still come from `Near` and `M2L`, but the new setup-side `panelIA0()` acceleration removes a large fraction of the previous one-time setup cost.

At `depth=5`, the GPU advantage is even larger in absolute speedup (`32.2x` for hybrid vs CPU), mainly because the CPU near-field cost is still extremely large there. As depth increases to `7` and `8`, the total speedup drops because more work shifts away from near-direct interactions and toward the FMM stages, especially `M2L`. This is consistent with the earlier observation that the best depth for GPU/hybrid is not the same tradeoff as for CPU serial.

One additional point is that the direct appendix is not really informative for `1a63`. The direct GPU baseline does not fit as a true dense direct run for this case size, so it is not a clean direct-vs-FMM comparison here. For the large protein, the meaningful comparison remains CPU serial vs GPU full vs hybrid FMM.

My current interpretation is:

- the new default implementation is clearly better than the previous one because the setup-side `panelIA0()` bottleneck has been reduced substantially
- for `1a63`, `depth=6` is the best point in the current sweep
- the adaptive parameter study should continue to focus on `nLev` and `SepRat`, since the GPU/CPU tradeoff is still controlled mainly by the balance between `Near` and `M2L`

If you want, I can next summarize the same matrix in a small table for the paper notes, and then continue the adaptive sweep with the improved default implementation.

Best,
[Your Name]
