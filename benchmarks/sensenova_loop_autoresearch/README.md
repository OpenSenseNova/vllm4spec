# Historical image-text autoresearch workload

This harness describes the fixed October 2026 experiment on `ckpt-225-bf16`,
with 80 execution layers and four residual streams. Its module and environment
variable references follow the current SenseNovaLoop naming, but its frozen
sample paths, reference captures, and measurement protocol still belong to that
historical experiment. The referenced image-text recipes and shell environment
are no longer present in the current SenseSpec checkout.

This is not a runnable benchmark for `SenseNova-6.8-Loop-9B-Preview-mtp13200`.
Use the maintained [text inference and benchmark example](../../examples/SenseNova-Loop-9B-Preview-mhc/README.md)
for that checkpoint. Do not substitute new weights into this harness and compare
the resulting timings or hashes against its historical baseline.
