## Runtime

Single node: 128 CPU cores, 240 GiB RAM, 0 GPUs. All calculations share these resources.

Whole-task limits: 5 hours and 5000 agent execution steps, whichever comes first. Check remaining time with `bench-runtime status`. All processes stop when the attempt ends.

Workspace: `/work/ws`. Save the requested output files there before finishing.

Launch software with `bench-exec TOOL --ranks N --threads T -- ARGS...`. Use `bench-exec --list` and `bench-exec --help` to inspect available commands. Further details: `/opt/bench/commands.md` and `/opt/bench/software.json`.
