# Engineering Standards

## Subagent Workflow

- One subagent, one dedicated worktree.
- Use a dedicated `codex/*` branch per subagent and keep the branch isolated from other agents.
- Do not develop directly on `master` or `main`.
- Before running task scripts, run `.\scripts\ensure_subagent_worktree.ps1` or let the entry script invoke it.
- Commit only after the relevant tests pass.
- If a task needs multiple iterations, keep going until the output is complete and testable.

## Code Quality Checklist

- Use descriptive names for files, functions, types, variables, and scripts.
- Add short comments for non-obvious logic, especially around parsing, validation, and numerical steps.
- Keep public interfaces append-only when compatibility matters.
- Add or update tests for every behavior change.
- Guard performance-sensitive paths with measurable checks or benchmarks when practical.
- Prefer small, cohesive helpers over large monolithic blocks.
- Keep error handling explicit and console-visible.

## Testing Policy

- Tests must print failures to the console, not through popup dialogs or debugger prompts.
- Favor deterministic, non-interactive test commands.
- Run targeted smoke tests before committing.
- If a test fails, fix the root cause rather than hiding it with suppressed output.

## Practical Guardrails

- Preserve backward compatibility for existing JSON and CLI inputs unless a breaking change is explicitly approved.
- Keep docs, scripts, and examples in sync with behavior changes.
- For user-facing workflow changes, update `README.md`, `docs/USAGE.md`, or `docs/TESTING.md` in the same change set.
