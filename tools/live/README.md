# Live operator tools

There are currently no verified active scripts in this directory. The former root-level live
runners depended on source modules that were never present in tracked history, so they are
preserved with explicit limitations under `legacy/incomplete-live-tools`.

A tool may be promoted here only after its imports resolve from tracked packages, external
effects require explicit execution acknowledgement and hard budgets, outputs use an external
run root, and focused offline tests pass.
