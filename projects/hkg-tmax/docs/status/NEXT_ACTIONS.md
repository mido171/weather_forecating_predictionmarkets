# Next actions and gates

1. Complete monorepo cutover verification and retain the locked rollback
   worktree through the observation window.
2. Verify external data and run-import manifests before removing any rollback
   copy.
3. Recreate the HKG virtual environment from `pyproject.toml` and run the full
   offline release suite once.
4. Rotate the externally managed proxy credential exposed by old history; local
   source neutralization does not revoke it.
5. Re-enable only individually reviewed collectors with explicit provider,
   request, byte, runtime, and retry budgets.
6. Continue scientific work only from a governed campaign experiment and the
   exact as-of/settlement contracts.
