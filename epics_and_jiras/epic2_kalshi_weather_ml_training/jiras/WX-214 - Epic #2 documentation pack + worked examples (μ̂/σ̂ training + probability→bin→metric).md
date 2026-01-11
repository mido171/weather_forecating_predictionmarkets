# WX-214 — Epic #2 documentation pack + worked examples (μ̂/σ̂ training + probability→bin→metric)

## Objective
Write clear, strict documentation so Epic #2 is implementable without guesswork.

## Required docs updates/additions
- `ml/README.md`
- `docs/data-contract.md` (confirm final)
- `docs/evaluation-metrics.md` (confirm final)
- `docs/model-persistence.md` (confirm final)
- Add `docs/worked-example.md` with 2 numeric examples:
  1) training row: residual → v(T) = r^2 → sigma training target
  2) inference: μ̂/σ̂ → integer probs → bin probs → Brier/log loss computation

## Acceptance Criteria
- [ ] A dev can run the training CLI end-to-end using only docs.
- [ ] Worked examples include explicit numbers and match implementation outputs.

