# Hit 18:30 Stockholm V5 Experiments Report

Always-NO test accuracy: 0.625

Always-YES test accuracy: 0.375

| Experiment | Val Acc | Val Bal Acc | Val YES Recall | Test Acc | Test Bal Acc | Test YES Recall | NetUnits/100 (Test, c0.55 ev0.03) | TradeRate (Test) |
|---|---|---|---|---|---|---|---|---|
| EXP01_Fusion_SBFD | 0.761 | 0.724 | 0.596 | 0.749 | 0.705 | 0.530 | 10.03 | 0.210 |
| EXP02_Fusion_REVSHOCK | 0.732 | 0.692 | 0.556 | 0.734 | 0.693 | 0.530 | 10.52 | 0.219 |
| EXP03_Fusion_BC_GAP | 0.752 | 0.714 | 0.582 | 0.751 | 0.715 | 0.572 | 12.68 | 0.223 |
| EXP04_Fusion_REG_BETA_CAL | 0.761 | 0.726 | 0.604 | 0.745 | 0.710 | 0.570 | 10.82 | 0.220 |
| EXP05_Fusion_EV_WEIGHTED | 0.756 | 0.716 | 0.580 | 0.750 | 0.709 | 0.543 | 11.90 | 0.219 |
| EXP06_Fusion_ANALOG_PRIOR | 0.746 | 0.708 | 0.577 | 0.746 | 0.705 | 0.541 | 11.01 | 0.218 |
| EXP07_HAZARD_EXCEED | 0.775 | 0.720 | 0.563 | 0.752 | 0.699 | 0.514 | 10.20 | 0.199 |
| EXP08_ZI_DELTA | 0.747 | 0.711 | 0.585 | 0.732 | 0.694 | 0.543 | 11.31 | 0.233 |
| EXP09_CNN2GBM | 0.745 | 0.703 | 0.559 | 0.735 | 0.697 | 0.549 | 9.34 | 0.225 |
| EXP10_CPC_EMBED | 0.753 | 0.715 | 0.585 | 0.748 | 0.712 | 0.567 | 10.42 | 0.226 |
| EXP11_MOE_4REG | 0.749 | 0.708 | 0.566 | 0.736 | 0.696 | 0.535 | 10.82 | 0.224 |
| EXP12_MONO_PHYS | 0.730 | 0.669 | 0.460 | 0.715 | 0.660 | 0.444 | 8.95 | 0.184 |
| EXP13_SLOPE_HIST | 0.748 | 0.704 | 0.553 | 0.737 | 0.701 | 0.554 | 9.73 | 0.233 |
| EXP14_DROP_REBOUND | 0.754 | 0.709 | 0.556 | 0.732 | 0.696 | 0.556 | 9.44 | 0.240 |
| EXP15_QA_AWARE | 0.767 | 0.727 | 0.590 | 0.732 | 0.694 | 0.546 | 10.42 | 0.240 |
| EXP16_CTP_BAYES | 0.758 | 0.714 | 0.561 | 0.731 | 0.689 | 0.522 | 9.54 | 0.223 |
| EXP17_MOS_NOWCAST_MIS_v2 | 0.755 | 0.710 | 0.556 | 0.735 | 0.696 | 0.538 | 10.23 | 0.230 |
| EXP18_EV_STACK_OOF | 0.732 | 0.684 | 0.519 | 0.729 | 0.687 | 0.522 | 9.44 | 0.212 |
| EXP19_MT_DELTA_BINS | 0.757 | 0.713 | 0.561 | 0.737 | 0.695 | 0.528 | 10.52 | 0.217 |
| EXP20_HB_PARAM_T | 0.747 | 0.704 | 0.556 | 0.728 | 0.690 | 0.541 | 9.44 | 0.208 |

Best by Val Accuracy: EXP07_HAZARD_EXCEED (Val Acc 0.775)

Best by Val Balanced Accuracy: EXP15_QA_AWARE (Val Bal Acc 0.727)

