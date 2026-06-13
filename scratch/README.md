# scratch/ — quarantined debug scripts

One-off diagnostic / investigation scripts (`diag_*`, `_*`) that are NOT needed to run
the model or build the webapp. Kept for reference, tracked in git, but not maintained.

**Caveat:** imports here may be stale — many predate the package reorg and still do
`import bench_utils` (now `oculomotor.benchmarks.bench_utils`) etc. Fix imports if you
revive one. Many are numbered iterations (`_diag_sg`/`sg2`/`sg3`) — dead duplicates to prune later.
