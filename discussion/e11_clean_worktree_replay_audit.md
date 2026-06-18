# E11 Clean Worktree Replay Audit

This generated audit replays the CPU-side artifact gate from a detached clean Git worktree. It is a reproducibility check for tracked files only: `serverREADME.md` is intentionally absent, and the command does not create new empirical evidence.

## Run Summary

| audit_id                        | status   | replayed_revision                        | source_branch_at_generation   | command                                                               |   elapsed_seconds | server_readme_present_in_worktree   | post_status_short   | post_untracked   |
|:--------------------------------|:---------|:-----------------------------------------|:------------------------------|:----------------------------------------------------------------------|------------------:|:------------------------------------|:--------------------|:-----------------|
| clean_worktree_e11_check_replay | pass     | dbcc73bc92b05b0a5299a45c97f3f58f6d9b4610 | codex/resnet-gpu-tail-results | make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-check |             41.74 | no                                  | clean               | none             |

## Gate Matrix

| gate_id                          | status   | evidence                                                        | claim_effect                                                       |
|:---------------------------------|:---------|:----------------------------------------------------------------|:-------------------------------------------------------------------|
| CWR-1-detached-worktree-created  | pass     | git worktree add --detach completed                             | tracked-source replay is available                                 |
| CWR-2-local-attachment-excluded  | pass     | serverREADME.md absent from clean worktree                      | local attachment is not part of the submitted evidence bundle      |
| CWR-3-e11-check-passes           | pass     | make e11-check exit_code=0                                      | validator, pytest, and whitespace checks replay from tracked files |
| CWR-4-pytest-pass-observed       | pass     | clean worktree log contains `71 passed`                         | test suite pass is visible in replay log                           |
| CWR-5-git-diff-check-observed    | pass     | clean worktree log contains `git diff --check`                  | whitespace check is part of the replayed gate                      |
| CWR-6-worktree-clean-after-check | pass     | git status --short and untracked list are empty after e11-check | the replay check does not mutate tracked evidence files            |

## Operating Rule

This audit supports tracked-source CPU replay of `make PYTHON=/data/conda_envs/SpatialQuantization/bin/python e11-check`. It does not close the preferred `pdflatex`/`bibtex`/`xelatex` clean-checkout gate and does not authorize stronger scientific claims.

Artifacts:
- [run_summary.csv](../results/e11_clean_worktree_replay_audit/run_summary.csv)
- [gate_matrix.csv](../results/e11_clean_worktree_replay_audit/gate_matrix.csv)
- [command_log_tail.txt](../results/e11_clean_worktree_replay_audit/command_log_tail.txt)
- [config.json](../results/e11_clean_worktree_replay_audit/config.json)
