# Hit 18:30 Stockholm V2 Experiments Report

Always-NO test accuracy: 0.625

Always-YES test accuracy: 0.375

| Experiment | Val Acc | Val Bal Acc | Val YES Recall | Test Acc | Test Bal Acc | Test YES Recall | NetUnits/100 (Val) | NetUnits/100 (Test) |
|---|---|---|---|---|---|---|---|---|
| EXP11_BASELINE_REPRO | 0.748 | 0.691 | 0.495 | 0.736 | 0.684 | 0.475 | 10.28 | 11.11 |
| EXP12_ADD_HEAT_GAP | 0.757 | 0.715 | 0.572 | 0.746 | 0.701 | 0.522 | 11.13 | 12.09 |
| EXP13_ADD_MOS_TBLOCK_MISMATCH | 0.746 | 0.682 | 0.460 | 0.731 | 0.669 | 0.423 | 10.09 | 10.52 |
| EXP14_ADD_PEAK_CONFIDENCE_V2 | 0.753 | 0.702 | 0.527 | 0.719 | 0.669 | 0.472 | 10.75 | 9.34 |
| EXP15_ADD_CONDITIONAL_CLIMO | 0.738 | 0.670 | 0.439 | 0.728 | 0.659 | 0.386 | 9.25 | 10.23 |
| EXP16_ADD_ANALOG_KNN_PRIOR | 0.745 | 0.683 | 0.468 | 0.733 | 0.666 | 0.402 | 10.00 | 10.72 |
| EXP19_REVISION_V2_MINIMAL | 0.743 | 0.714 | 0.614 | 0.723 | 0.684 | 0.528 | 9.81 | 9.73 |
| EXP17_MONOTONIC_CONSTRAINTS | 0.746 | 0.688 | 0.489 | 0.728 | 0.665 | 0.415 | 10.09 | 10.23 |
| EXP18_SEASONAL_2MODEL | 0.748 | 0.672 | 0.410 | 0.716 | 0.639 | 0.333 | 10.28 | 9.05 |
| EXP20_PROPER_OOF_STACK | 0.720 | 0.650 | 0.410 | 0.710 | 0.639 | 0.357 | 7.45 | 8.46 |

Best by Val Accuracy: EXP12_ADD_HEAT_GAP (Val Acc 0.757)

Best by Val Balanced Accuracy: EXP12_ADD_HEAT_GAP (Val Bal Acc 0.715)

Overfit diagnostics (val-test acc gap > 0.08):
