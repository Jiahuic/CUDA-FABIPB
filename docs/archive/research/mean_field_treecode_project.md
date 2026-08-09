# FMM-Inspired Mean-Field Acceleration

This note records a possible next project direction after the current GPU-FABIPB
work: fast many-body acceleration for mean-field models over biological
variants or sequence embeddings.

## Core Model

Many mean-field models can be written as a dense interaction:

```text
F_i = sum_j K(x_i, x_j) q_j
```

where:

- `x_i` is a variant, sequence, structure, or learned embedding
- `q_j` is the source weight, state, or inferred field variable
- `K(x_i, x_j)` is an interaction kernel in embedding or feature space
- `F_i` is the resulting mean field acting on item `i`

A direct evaluation costs `O(N^2)`, which becomes impractical for large variant
sets.

## Why Start With Treecode

A treecode is a better first fit than full FMM for this mean-field direction.

Treecode keeps the core mean-field interpretation clear:

- nearby variants interact directly
- far clusters contribute through aggregate summaries
- each target receives an approximate field from cluster-level statistics

This maps naturally to mean-field theory because a far cluster is replaced by
its collective effect. Full FMM can come later if the treecode approximation is
accurate but still not fast enough.

## Proposed Treecode Formulation

1. Embed variants into a metric space:
   - sequence embedding
   - structure-aware embedding
   - learned latent representation

2. Build a hierarchical tree over embeddings:
   - kd-tree, ball tree, cover tree, or octree-like partition
   - split until each leaf has a bounded number of variants

3. Store cluster summaries:
   - cluster center or centroid
   - total source strength `sum q_j`
   - low-order moments of `q_j` around the centroid
   - optional covariance or anisotropic shape summary

4. Evaluate the field:
   - if a source cluster is near the target, descend or use direct interactions
   - if a source cluster is well separated, approximate its contribution using
     the cluster summary

5. Validate against direct `O(N^2)` evaluation on small cases.

## Acceptance Criterion

For a target point `x_i` and source cluster `C`, accept the cluster approximation
when:

```text
radius(C) / distance(x_i, center(C)) < theta
```

where `theta` controls the accuracy/speed tradeoff.

This gives a simple first version:

```text
F_i ~= sum_near K(x_i, x_j) q_j
     + sum_far K(x_i, c_C) Q_C
```

with:

```text
Q_C = sum_{j in C} q_j
```

Higher-order moment corrections can be added after the zeroth-order treecode is
validated.

## GPU-Relevant Work

The GPU path should target:

- batched direct interactions for near leaves
- batched target-cluster evaluations for accepted far clusters
- parallel construction/update of cluster summaries
- repeated mean-field iterations where the tree structure is fixed but `q_j`
  changes

This is close to the current GPU-FABIPB experience:

- near interactions resemble GPU near-field kernels
- cluster summaries resemble multipole data
- tree traversal and batching resemble FMM scheduling
- accuracy/runtime tuning resembles `nLev`, `SepRat`, and near/far policy

## Research Positioning

This project is not pure mean field and not pure FMM.

The right framing is:

```text
mean-field modeling plus fast many-body computation
```

The treecode version is the most natural first paper prototype because it keeps
the biological/statistical mean-field interpretation visible while still using
hierarchical acceleration.

## First Implementation Target

Build a small standalone prototype before integrating with any biology-specific
pipeline:

1. Input: embeddings `X`, source weights `q`, kernel choice, and `theta`.
2. Baseline: direct dense evaluation.
3. Approximation: treecode evaluation.
4. Output: runtime, relative error, and accepted-cluster statistics.
5. Later extension: replace zeroth-order summaries with learned or analytic
   multipole-like summaries.

