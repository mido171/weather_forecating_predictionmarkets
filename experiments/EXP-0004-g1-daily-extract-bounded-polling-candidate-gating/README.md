# EXP-0004 - G1 Daily Extract bounded polling candidate gating

**Status:** ACCEPTED  
**Created:** 2026-06-18T17:33:58.120677Z  
**Goal dependency:** G1  
**Owner:** Codex

## Plain-language question

Can the Daily Extract poller run a bounded repeat loop and still prevent
provider-first-publication candidate labels except for explicitly watched target
dates?

## Why this could improve Tmax

EXP-0003 proved a one-shot first-observation ledger. G1 still needs repeated
near-publication polling. This experiment adds a safer operational loop and
candidate-date gate so future first-publication evidence can accumulate without
mislabeling already-published historical rows.

## What changed from prior work

EXP-0003 accepted polling mechanics for archive-first-observed evidence.
EXP-0004 adds bounded repeat polling and explicit watched-date candidate
eligibility. It does not perform modelling, machine learning, or market
backtesting.

## Current conclusion

Accepted as polling safety infrastructure. G1 remains blocked pending provider
first-publication evidence.
