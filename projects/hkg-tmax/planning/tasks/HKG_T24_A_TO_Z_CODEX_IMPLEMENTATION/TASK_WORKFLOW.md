# A-to-Z Task Workflow

Package-level files in this directory are reference material, contracts, specs, schemas, evidence, tools, and source archives. They are not task-completion state.

Task state lives under:

- `tasks/not-completed/Txx_suffix/` for unfinished tasks.
- `tasks/completed/Txx_suffix/` for finished tasks.

Each completed task folder contains its original task document, copied task spec, and `COMPLETION_RECORD.md`. Implementation evidence lives in the corresponding numbered folder under repository `experiments/`.

`TASK_STATUS_INDEX.csv` is the current task-state index. `TASK_INDEX.csv` remains the original dependency DAG from the source bundle.
