# Production Eligibility Gate

Live trading is disabled by default. Every item below must be signed off.

## A. Target and rules

- [ ] exact source/field/date/precision proven;
- [ ] first-publication archive operational;
- [ ] CLMMAXT parity quantified;
- [ ] computed winner matches all verifiable resolved events;
- [ ] rules hash monitor tested;
- [ ] unknown/changed rules halt system.

## B. Point-in-time data

- [ ] every operational source has an availability contract;
- [ ] raw archive is immutable and monitored;
- [ ] actual model file delivery latency is enforced;
- [ ] no retrospective-only feature enters production;
- [ ] missing/stale feeds cause fallback or no-trade;
- [ ] source schemas and units are tested.

## C. Forecast validation

- [ ] primary horizon frozen;
- [ ] strong baselines established;
- [ ] champion passes locked test;
- [ ] leakage audit PASS;
- [ ] reproducibility review PASS;
- [ ] calibration and tail diagnostics pass;
- [ ] live shadow sample spans multiple regimes;
- [ ] predictions are timestamped and immutable.

## D. Market execution research

- [ ] exact token/outcome mapping tested;
- [ ] per-market fees queried and stored;
- [ ] live book sequence/reconnect handling tested;
- [ ] stale-book detection tested;
- [ ] conservative fill model validated against paper orders;
- [ ] spread/slippage sensitivity passes;
- [ ] inventory aggregation across outcomes works;
- [ ] no midpoint-only P&L claims.

## E. Risk limits

- [ ] event loss cap;
- [ ] daily loss cap;
- [ ] per-bucket inventory cap;
- [ ] aggregate correlated exposure cap;
- [ ] minimum net-edge threshold;
- [ ] uncertainty/disagreement no-trade threshold;
- [ ] source anomaly kill switch;
- [ ] rules/source change kill switch;
- [ ] model drift kill switch;
- [ ] manual emergency stop.

## F. Operations

- [ ] monitoring dashboard;
- [ ] alert routing;
- [ ] audit log;
- [ ] deterministic forecast rerun;
- [ ] clock synchronization;
- [ ] backup/archive integrity;
- [ ] disaster recovery;
- [ ] runbook drill;
- [ ] no secret in logs or Git.

## Sign-off

```yaml
target_parity_reviewer:
leakage_reviewer:
reproducibility_reviewer:
market_execution_reviewer:
risk_owner:
date:
effective_model_version:
effective_rules_hashes:
```

Until all sections pass, outputs are research or paper-trading only.
