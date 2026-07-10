# Results

Generated: `2026-07-07T22:47:23Z`

## Headline

Significance score: **82/100**.

This is a meaningful station-hour signal discovery result, not a deployable champion. The strongest signals are real and meteorologically coherent, but the guarded one-feature residual corrections are small after the official forecast anchor and bias correction are accounted for.

Best guarded univariate residual correction: `hko__latest_temp_minus_official_min_c`; candidate MAE `0.919812` vs official `0.927334` and bias-only `0.922780`. Delta vs bias-only `-0.002969` C.

## Data Scope

| Metric | Value |
|---|---:|
| Frame rows | 8762 |
| Frame dates | 2000-01-02 to 2023-12-31 |
| Hourly rows joined | 228391 |
| Station-long rows joined | 4620370 |
| Feature-value rows | 7220809 |
| Distinct features | 1083 |
| Distinct stations | 27 |
| Confirmation rows used | 0 |

## Top Pearson Signals

| feature_name | feature_family | station | station_role | transform | window_hours | snapshot_hour | n | max_abs_primary_corr | pearson_residual | pearson_abs_error | pearson_under_gt1 | pearson_over_gt1 | pearson_hot_under | residual_corr_train_eval_same_sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| station_tai_mei_tuk__h18_temp_c | station_snapshot | TAI MEI TUK | inland_nt | snapshot_temp |  | 18 | 2537 | 0.26443 | -0.0420758 | -0.0720342 | -0.073051 | -0.0177957 | 0.26443 | False |
| station_tai_mei_tuk__w6h_temp_max_c | station_window | TAI MEI TUK | inland_nt | temp_max | 6 |  | 2541 | 0.263044 | -0.0383983 | -0.0703895 | -0.06805 | -0.0193107 | 0.263044 | False |
| station_kai_tak_runway_park__h15_temp_c | station_snapshot | KAI TAK RUNWAY PARK | urban_core | snapshot_temp |  | 15 | 3252 | 0.260452 | -0.0410498 | -0.0907259 | -0.0787167 | -0.0200247 | 0.260452 | False |
| station_tai_mei_tuk__w12h_temp_mean_c | station_window | TAI MEI TUK | inland_nt | temp_mean | 12 |  | 2542 | 0.260397 | -0.0387422 | -0.0767589 | -0.0729487 | -0.0220865 | 0.260397 | False |
| station_tai_mei_tuk__h15_temp_c | station_snapshot | TAI MEI TUK | inland_nt | snapshot_temp |  | 15 | 2535 | 0.260209 | -0.0343826 | -0.0894557 | -0.0788477 | -0.0345505 | 0.260209 | False |
| station_tai_mei_tuk__w12h_temp_max_c | station_window | TAI MEI TUK | inland_nt | temp_max | 12 |  | 2542 | 0.259292 | -0.0413208 | -0.0891456 | -0.0816664 | -0.0293218 | 0.259292 | False |
| station_tai_mei_tuk__w24h_temp_max_c | station_window | TAI MEI TUK | inland_nt | temp_max | 24 |  | 2545 | 0.256808 | -0.0414771 | -0.0795397 | -0.0805517 | -0.022866 | 0.256808 | False |
| station_kai_tak_runway_park__w12h_temp_max_c | station_window | KAI TAK RUNWAY PARK | urban_core | temp_max | 12 |  | 3278 | 0.254399 | -0.0427408 | -0.0896649 | -0.0797478 | -0.0187291 | 0.254399 | False |
| station_kai_tak_runway_park__h18_temp_c | station_snapshot | KAI TAK RUNWAY PARK | urban_core | snapshot_temp |  | 18 | 3265 | 0.253865 | -0.0368103 | -0.0872125 | -0.0733104 | -0.0198275 | 0.253865 | False |
| station_kai_tak_runway_park__w24h_temp_max_c | station_window | KAI TAK RUNWAY PARK | urban_core | temp_max | 24 |  | 3280 | 0.253402 | -0.04564 | -0.083461 | -0.0799198 | -0.012616 | 0.253402 | False |
| station_kai_tak_runway_park__w6h_temp_max_c | station_window | KAI TAK RUNWAY PARK | urban_core | temp_max | 6 |  | 3277 | 0.25075 | -0.0388583 | -0.0823137 | -0.0730995 | -0.0133724 | 0.25075 | False |
| station_tai_mei_tuk__w6h_temp_mean_c | station_window | TAI MEI TUK | inland_nt | temp_mean | 6 |  | 2541 | 0.249518 | -0.037125 | -0.0670996 | -0.0673914 | -0.0163197 | 0.249518 | False |
| station_kai_tak_runway_park__w12h_temp_mean_c | station_window | KAI TAK RUNWAY PARK | urban_core | temp_mean | 12 |  | 3278 | 0.249233 | -0.0403998 | -0.087275 | -0.0764013 | -0.017457 | 0.249233 | False |
| station_tai_mei_tuk__w24h_temp_mean_c | station_window | TAI MEI TUK | inland_nt | temp_mean | 24 |  | 2545 | 0.248106 | -0.0514957 | -0.0650258 | -0.0784537 | -0.00422613 | 0.248106 | False |
| station_tai_mei_tuk__h21_temp_c | station_snapshot | TAI MEI TUK | inland_nt | snapshot_temp |  | 21 | 2537 | 0.242837 | -0.035582 | -0.0643464 | -0.0668362 | -0.0149554 | 0.242837 | False |
| station_yuen_long_park__h18_temp_c | station_snapshot | YUEN LONG PARK | inland_nt | snapshot_temp |  | 18 | 3196 | 0.242789 | -0.0345038 | -0.0855567 | -0.0735621 | -0.0183274 | 0.242789 | False |
| station_yuen_long_park__w6h_temp_max_c | station_window | YUEN LONG PARK | inland_nt | temp_max | 6 |  | 3203 | 0.242031 | -0.0356614 | -0.0828086 | -0.0725895 | -0.0159221 | 0.242031 | False |
| station_tai_mei_tuk__w12h_temp_min_c | station_window | TAI MEI TUK | inland_nt | temp_min | 12 |  | 2542 | 0.239026 | -0.0360825 | -0.0665016 | -0.0668848 | -0.0147455 | 0.239026 | False |
| station_kai_tak_runway_park__w6h_temp_mean_c | station_window | KAI TAK RUNWAY PARK | urban_core | temp_mean | 6 |  | 3277 | 0.238627 | -0.0385979 | -0.0821838 | -0.0731779 | -0.0137582 | 0.238627 | False |
| station_kai_tak_runway_park__w24h_temp_mean_c | station_window | KAI TAK RUNWAY PARK | urban_core | temp_mean | 24 |  | 3280 | 0.238267 | -0.0515232 | -0.0793475 | -0.0817785 | -0.00284217 | 0.238267 | False |
| station_tai_mei_tuk__w6h_temp_min_c | station_window | TAI MEI TUK | inland_nt | temp_min | 6 |  | 2541 | 0.237927 | -0.0343284 | -0.0666696 | -0.0661753 | -0.0167113 | 0.237927 | False |
| station_yuen_long_park__w12h_temp_mean_c | station_window | YUEN LONG PARK | inland_nt | temp_mean | 12 |  | 3203 | 0.23772 | -0.0407405 | -0.0839351 | -0.0760431 | -0.0130852 | 0.23772 | False |
| station_yuen_long_park__w24h_temp_max_c | station_window | YUEN LONG PARK | inland_nt | temp_max | 24 |  | 3203 | 0.235735 | -0.0379553 | -0.0856515 | -0.0790126 | -0.0169796 | 0.235735 | False |
| station_yuen_long_park__h15_temp_c | station_snapshot | YUEN LONG PARK | inland_nt | snapshot_temp |  | 15 | 3189 | 0.235537 | -0.0391702 | -0.0991725 | -0.0828509 | -0.0274249 | 0.235537 | False |
| station_yuen_long_park__w12h_temp_max_c | station_window | YUEN LONG PARK | inland_nt | temp_max | 12 |  | 3203 | 0.235246 | -0.0375749 | -0.0916956 | -0.0800096 | -0.0219138 | 0.235246 | False |

## Top Spearman And Quantile-Spread Signals

| feature_name | n_values | spearman_residual | spearman_abs_error | spearman_under_gt1 | spearman_over_gt1 | spearman_hot_under | q10_q90_response_spread | low_n | high_n | max_abs_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hko__latest_temp_minus_official_min_c | 8760 | 0.150491 | -0.0330231 |  |  |  | 0.479689 | 4450 | 1585 | 0.150491 |
| network__w6h_max_minus_official_max_c | 8748 | 0.145174 | 0.038462 |  |  |  | 0.506197 | 2240 | 1881 | 0.145174 |
| role_inland_nt__w6h_max_minus_official_max_c | 8672 | 0.144919 | 0.0223027 |  |  |  | 0.623953 | 1590 | 1152 | 0.144919 |
| station_sha_tin__w6h_max_minus_official_max_c | 8738 | 0.142512 | 0.0269792 |  |  |  | 0.595911 | 1082 | 1582 | 0.142512 |
| hko__latest_temp_minus_official_max_c | 8760 | 0.138915 | 0.026468 |  |  |  | 0.557048 | 2499 | 1652 | 0.138915 |
| role_coastal_marine__w6h_max_minus_official_max_c | 8752 | 0.138892 | 0.0375376 |  |  |  | 0.47592 | 2660 | 1657 | 0.138892 |
| station_sha_tin__h18_minus_official_max_c | 8726 | 0.138806 | 0.0198141 |  |  |  | 0.567723 | 1233 | 1505 | 0.138806 |
| network__latest_max_minus_official_max_c | 8760 | 0.137992 | 0.0658323 |  |  |  | 0.646068 | 1271 | 1348 | 0.137992 |
| station_king_s_park__latest_minus_official_max_c | 8760 | 0.137283 | 0.048064 |  |  |  | 0.678121 | 1980 | 905 | 0.137283 |
| station_king_s_park__h23_minus_official_max_c | 8741 | 0.137031 | 0.048164 |  |  |  | 0.676131 | 1978 | 901 | 0.137031 |
| station_chek_lap_kok__h23_minus_official_max_c | 8746 | 0.135812 | 0.0564333 |  |  |  | 0.65931 | 1619 | 1090 | 0.135812 |
| station_chek_lap_kok__latest_minus_official_max_c | 8760 | 0.135713 | 0.0558517 |  |  |  | 0.657285 | 1620 | 1092 | 0.135713 |
| station_shek_kong__w6h_max_minus_official_max_c | 7498 | 0.135043 | 0.0313039 |  |  |  | 0.549371 | 1923 | 810 | 0.135043 |
| station_shek_kong__h18_minus_official_max_c | 7492 | 0.133592 | 0.0257478 |  |  |  | 0.576293 | 769 | 789 | 0.133592 |
| station_wong_chuk_hang__w6h_max_minus_official_max_c | 8700 | 0.133076 | 0.0544999 |  |  |  | 0.614805 | 906 | 1692 | 0.133076 |
| station_wong_chuk_hang__h18_minus_official_max_c | 8682 | 0.132437 | 0.0483701 |  |  |  | 0.604876 | 999 | 1587 | 0.132437 |
| station_chek_lap_kok__w6h_max_minus_official_max_c | 8758 | 0.130703 | 0.0267466 |  |  |  | 0.568762 | 1034 | 1512 | 0.130703 |
| station_ta_kwu_ling__w6h_max_minus_official_max_c | 8721 | 0.129577 | -0.00932533 |  |  |  | 0.537421 | 1360 | 1550 | 0.129577 |
| station_ta_kwu_ling__h18_minus_official_max_c | 8706 | 0.128847 | -0.0105396 |  |  |  | 0.528417 | 1428 | 1530 | 0.128847 |
| station_king_s_park__w6h_max_minus_official_max_c | 8754 | 0.12713 | 0.016212 |  |  |  | 0.527162 | 1573 | 1091 | 0.12713 |
| station_chek_lap_kok__h21_minus_official_max_c | 8752 | 0.126914 | 0.0407926 |  |  |  | 0.571103 | 1131 | 1522 | 0.126914 |
| station_chek_lap_kok__h18_minus_official_max_c | 8747 | 0.12456 | 0.0140779 |  |  |  | 0.509632 | 1270 | 1412 | 0.12456 |
| station_tsing_yi__w6h_max_minus_official_max_c | 7764 | 0.124052 | 0.0433652 |  |  |  | 0.497351 | 930 | 1537 | 0.124052 |
| station_king_s_park__h18_minus_official_max_c | 8743 | 0.122859 | 0.00809316 |  |  |  | 0.490236 | 1772 | 1025 | 0.122859 |
| station_king_s_park__h21_minus_official_max_c | 8748 | 0.122605 | 0.039802 |  |  |  | 0.555425 | 1622 | 1156 | 0.122605 |

## Feature-Family Summary

| feature_family | feature_count | stable_train_eval_count | best_feature_name | best_max_abs_primary_corr | best_pearson_residual | best_pearson_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| station_snapshot | 216 | 160 | station_tai_mei_tuk__h18_temp_c | 0.26443 | -0.0420758 | -0.0720342 |
| station_window | 486 | 359 | station_tai_mei_tuk__w6h_temp_max_c | 0.263044 | -0.0383983 | -0.0703895 |
| station_latest | 54 | 37 | station_tai_mei_tuk__latest_temp_c | 0.23403 | -0.0345697 | -0.0639128 |
| role_window | 48 | 45 | role_urban_core__w6h_temp_max_c | 0.220746 | -0.0421174 | -0.08198 |
| hko_window | 21 | 21 | hko__w6h_temp_max_c | 0.19113 | -0.0662179 | -0.09887 |
| network_window | 18 | 18 | network__w6h_temp_max_c | 0.190057 | -0.058385 | -0.0890854 |
| forecast_contradiction | 234 | 202 | station_tsuen_wan__w6h_max_minus_official_max_c | 0.180188 | 0.180188 | 0.0931332 |
| hko_latest | 2 | 2 | hko__latest_temp_c | 0.175789 | -0.0607827 | -0.0975175 |
| network_latest | 4 | 4 | network__latest_max_c | 0.170195 | -0.0600892 | -0.0837035 |

## Station Leaderboard

| station | station_role | feature_name | feature_family | transform | window_hours | snapshot_hour | n | max_abs_primary_corr | pearson_residual | pearson_abs_error | pearson_under_gt1 | pearson_over_gt1 | pearson_hot_under | residual_corr_train_eval_same_sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TAI MEI TUK | inland_nt | station_tai_mei_tuk__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 2537 | 0.26443 | -0.0420758 | -0.0720342 | -0.073051 | -0.0177957 | 0.26443 | False |
| KAI TAK RUNWAY PARK | urban_core | station_kai_tak_runway_park__h15_temp_c | station_snapshot | snapshot_temp |  | 15 | 3252 | 0.260452 | -0.0410498 | -0.0907259 | -0.0787167 | -0.0200247 | 0.260452 | False |
| YUEN LONG PARK | inland_nt | station_yuen_long_park__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 3196 | 0.242789 | -0.0345038 | -0.0855567 | -0.0735621 | -0.0183274 | 0.242789 | False |
| KWUN TONG | urban_core | station_kwun_tong__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 5150 | 0.227865 | -0.0387048 | -0.080372 | -0.0600064 | -0.0145215 | 0.227865 | True |
| WONG TAI SIN | urban_core | station_wong_tai_sin__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 5360 | 0.224281 | -0.0355838 | -0.080576 | -0.056004 | -0.0162556 | 0.224281 | True |
| HAPPY VALLEY | urban_core | station_happy_valley__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 5478 | 0.22316 | -0.0437904 | -0.0818055 | -0.0611333 | -0.0100828 | 0.22316 | True |
| SHAU KEI WAN | coastal_marine | station_shau_kei_wan__h15_temp_c | station_snapshot | snapshot_temp |  | 15 | 5872 | 0.219588 | -0.0555565 | -0.106779 | -0.0780767 | -0.0169694 | 0.219588 | True |
| TSUEN WAN SHING MUN VALLEY | west_nw_nt | station_tsuen_wan_shing_mun_valley__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 4742 | 0.218719 | -0.0424647 | -0.0705548 | -0.0565876 | -0.00883844 | 0.218719 | False |
| KOWLOON CITY | urban_core | station_kowloon_city__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 5703 | 0.218104 | -0.0410379 | -0.083983 | -0.0593212 | -0.0141558 | 0.218104 | True |
| TSUEN WAN HO KOON | west_nw_nt | station_tsuen_wan_ho_koon__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 4735 | 0.217206 | -0.0402209 | -0.0697802 | -0.0545682 | -0.0107555 | 0.217206 | False |
| STANLEY | coastal_marine | station_stanley__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 5250 | 0.215154 | -0.0420598 | -0.0823133 | -0.0631168 | -0.0142296 | 0.215154 | True |
| SHAM SHUI PO | urban_core | station_sham_shui_po__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 5006 | 0.214781 | -0.0417172 | -0.0835772 | -0.0619904 | -0.0148869 | 0.214781 | True |
| HONG KONG PARK | urban_core | station_hong_kong_park__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 5840 | 0.208108 | -0.0574394 | -0.0937589 | -0.0730517 | -0.00728143 | 0.208108 | True |
| TAI PO | inland_nt | station_tai_po__h15_temp_c | station_snapshot | snapshot_temp |  | 15 | 8662 | 0.198089 | -0.0646449 | -0.109808 | -0.0852033 | -0.0163553 | 0.198089 | True |
| TA KWU LING | inland_nt | station_ta_kwu_ling__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 8706 | 0.197066 | -0.0547616 | -0.0985506 | -0.0726934 | -0.0145373 | 0.197066 | True |
| TSEUNG KWAN O | coastal_marine | station_tseung_kwan_o__h15_temp_c | station_snapshot | snapshot_temp |  | 15 | 8471 | 0.196467 | -0.0637518 | -0.113825 | -0.0869052 | -0.0166757 | 0.196467 | True |
| CHEUNG CHAU | coastal_marine | station_cheung_chau__h15_temp_c | station_snapshot | snapshot_temp |  | 15 | 8669 | 0.193608 | -0.0637169 | -0.118432 | -0.0908598 | -0.0216818 | 0.193608 | True |
| SAI KUNG | coastal_marine | station_sai_kung__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 8726 | 0.193001 | -0.0639201 | -0.105541 | -0.0815764 | -0.0117378 | 0.193001 | True |
| SHA TIN | inland_nt | station_sha_tin__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 8726 | 0.192035 | -0.0569578 | -0.0928491 | -0.0727682 | -0.00865397 | 0.192035 | True |
| SHEK KONG | inland_nt | station_shek_kong__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 7492 | 0.192024 | -0.0488366 | -0.0869612 | -0.0635905 | -0.00978863 | 0.192024 | True |
| KING'S PARK | urban_core | station_king_s_park__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 8743 | 0.190574 | -0.0639628 | -0.0975244 | -0.0794075 | -0.00633059 | 0.190574 | True |
| LAU FAU SHAN | west_nw_nt | station_lau_fau_shan__w6h_temp_max_c | station_window | temp_max | 6 |  | 8683 | 0.186952 | -0.0625649 | -0.0946331 | -0.0791209 | -0.00604296 | 0.186952 | True |
| TSING YI | west_nw_nt | station_tsing_yi__w12h_temp_max_c | station_window | temp_max | 12 |  | 7771 | 0.186128 | -0.0570525 | -0.0992212 | -0.0782937 | -0.0120347 | 0.186128 | True |
| WONG CHUK HANG | coastal_marine | station_wong_chuk_hang__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 8682 | 0.18403 | -0.061399 | -0.0915158 | -0.0756246 | -0.00427052 | 0.18403 | True |
| TUEN MUN | west_nw_nt | station_tuen_mun__h15_temp_c | station_snapshot | snapshot_temp |  | 15 | 8690 | 0.182494 | -0.0690679 | -0.106712 | -0.0903142 | -0.00902288 | 0.182494 | True |
| TSUEN WAN | west_nw_nt | station_tsuen_wan__w6h_max_minus_official_max_c | forecast_contradiction | window_max_minus_official_max | 6 |  | 1685 | 0.180188 | 0.180188 | 0.0931332 | 0.154514 | -0.0940936 | -0.0134913 | False |
| CHEK LAP KOK | coastal_marine | station_chek_lap_kok__h18_temp_c | station_snapshot | snapshot_temp |  | 18 | 8747 | 0.1793 | -0.0625104 | -0.0960449 | -0.0784242 | -0.00685514 | 0.1793 | True |

## Guarded Single-Feature Walk-Forward Actionability

| feature_name | folds | n_valid | official_mae | bias_only_mae | candidate_mae | delta_vs_official_c | delta_vs_bias_only_c | mean_abs_correction_c | folds_beating_official | folds_beating_bias_only | fold_deltas_vs_bias_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hko__latest_temp_minus_official_min_c | 4 | 4745 | 0.927334 | 0.92278 | 0.919812 | -0.00752237 | -0.00296861 | 0.183997 | 4 | 3 | fold1_2011_2013:-0.00772;fold2_2014_2016:0.00107;fold3_2017_2019:-0.00131;fold4_2020_2023:-0.00367 |
| station_tsing_yi__w24h_temp_trend_c | 4 | 4737 | 0.927127 | 0.922835 | 0.921061 | -0.00606605 | -0.00177385 | 0.179092 | 3 | 3 | fold1_2011_2013:-0.00779;fold2_2014_2016:0.01087;fold3_2017_2019:-0.00702;fold4_2020_2023:-0.00279 |
| station_tuen_mun__w24h_temp_min_c | 4 | 4732 | 0.927768 | 0.923197 | 0.921427 | -0.00634141 | -0.00176982 | 0.145029 | 4 | 3 | fold1_2011_2013:-0.00741;fold2_2014_2016:0.00491;fold3_2017_2019:-0.00355;fold4_2020_2023:-0.00123 |
| station_tseung_kwan_o__w24h_temp_min_c | 4 | 4723 | 0.925471 | 0.920758 | 0.919121 | -0.00634986 | -0.00163699 | 0.141869 | 4 | 2 | fold1_2011_2013:-0.00779;fold2_2014_2016:0.00463;fold3_2017_2019:-0.00404;fold4_2020_2023:0.00005 |
| station_sha_tin__w24h_temp_min_c | 4 | 4739 | 0.927137 | 0.922494 | 0.920972 | -0.00616487 | -0.00152253 | 0.138623 | 4 | 3 | fold1_2011_2013:-0.00820;fold2_2014_2016:0.00489;fold3_2017_2019:-0.00314;fold4_2020_2023:-0.00010 |
| station_lau_fau_shan__w24h_temp_min_c | 4 | 4709 | 0.927628 | 0.922987 | 0.921469 | -0.00615887 | -0.00151747 | 0.143088 | 4 | 3 | fold1_2011_2013:-0.00804;fold2_2014_2016:0.00564;fold3_2017_2019:-0.00329;fold4_2020_2023:-0.00068 |
| station_chek_lap_kok__w24h_temp_min_c | 4 | 4745 | 0.927334 | 0.922776 | 0.921452 | -0.00588155 | -0.00132369 | 0.138975 | 4 | 3 | fold1_2011_2013:-0.00787;fold2_2014_2016:0.00577;fold3_2017_2019:-0.00292;fold4_2020_2023:-0.00053 |
| station_tseung_kwan_o__w12h_temp_min_c | 4 | 4712 | 0.926125 | 0.921063 | 0.919839 | -0.006286 | -0.00122394 | 0.14221 | 4 | 2 | fold1_2011_2013:-0.00677;fold2_2014_2016:0.00329;fold3_2017_2019:-0.00248;fold4_2020_2023:0.00047 |
| station_tseung_kwan_o__w6h_temp_min_c | 4 | 4705 | 0.925824 | 0.920722 | 0.919503 | -0.00632052 | -0.00121912 | 0.142078 | 4 | 2 | fold1_2011_2013:-0.00685;fold2_2014_2016:0.00314;fold3_2017_2019:-0.00235;fold4_2020_2023:0.00056 |
| station_tai_po__w24h_temp_min_c | 4 | 4726 | 0.927973 | 0.923354 | 0.922165 | -0.00580839 | -0.00118982 | 0.143197 | 4 | 3 | fold1_2011_2013:-0.00702;fold2_2014_2016:0.00523;fold3_2017_2019:-0.00293;fold4_2020_2023:-0.00034 |
| station_sai_kung__w24h_temp_min_c | 4 | 4737 | 0.927 | 0.922428 | 0.921409 | -0.00559153 | -0.00101975 | 0.140349 | 4 | 3 | fold1_2011_2013:-0.00756;fold2_2014_2016:0.00642;fold3_2017_2019:-0.00284;fold4_2020_2023:-0.00030 |
| station_cheung_chau__w24h_temp_min_c | 4 | 4726 | 0.927444 | 0.922958 | 0.921966 | -0.00547806 | -0.000992321 | 0.141271 | 3 | 3 | fold1_2011_2013:-0.00707;fold2_2014_2016:0.00707;fold3_2017_2019:-0.00337;fold4_2020_2023:-0.00068 |
| station_king_s_park__w24h_temp_min_c | 4 | 4744 | 0.927319 | 0.92273 | 0.921745 | -0.00557369 | -0.000984798 | 0.140383 | 3 | 3 | fold1_2011_2013:-0.00805;fold2_2014_2016:0.00708;fold3_2017_2019:-0.00327;fold4_2020_2023:-0.00002 |
| station_tseung_kwan_o__w6h_temp_mean_c | 4 | 4705 | 0.925824 | 0.920722 | 0.919776 | -0.00604804 | -0.000946645 | 0.14181 | 4 | 2 | fold1_2011_2013:-0.00694;fold2_2014_2016:0.00424;fold3_2017_2019:-0.00229;fold4_2020_2023:0.00064 |
| station_tsing_yi__w24h_temp_mean_c | 4 | 4737 | 0.927127 | 0.922835 | 0.921901 | -0.00522538 | -0.000933184 | 0.143095 | 4 | 2 | fold1_2011_2013:-0.00618;fold2_2014_2016:0.00461;fold3_2017_2019:-0.00300;fold4_2020_2023:0.00042 |
| station_tseung_kwan_o__h21_temp_c | 4 | 4700 | 0.926149 | 0.920948 | 0.920039 | -0.00610977 | -0.000909236 | 0.142493 | 4 | 2 | fold1_2011_2013:-0.00647;fold2_2014_2016:0.00427;fold3_2017_2019:-0.00253;fold4_2020_2023:0.00057 |
| hko__w24h_temp_min_c | 4 | 4745 | 0.927334 | 0.922776 | 0.92197 | -0.00536427 | -0.000806413 | 0.142103 | 3 | 3 | fold1_2011_2013:-0.00781;fold2_2014_2016:0.00774;fold3_2017_2019:-0.00291;fold4_2020_2023:-0.00039 |
| station_lau_fau_shan__w24h_temp_mean_c | 4 | 4709 | 0.927628 | 0.922987 | 0.922365 | -0.00526263 | -0.00062123 | 0.141507 | 4 | 3 | fold1_2011_2013:-0.00557;fold2_2014_2016:0.00588;fold3_2017_2019:-0.00287;fold4_2020_2023:-0.00012 |
| role_west_nw_nt__w24h_temp_mean_c | 4 | 4745 | 0.927334 | 0.922771 | 0.922171 | -0.00516287 | -0.000600064 | 0.14315 | 4 | 3 | fold1_2011_2013:-0.00596;fold2_2014_2016:0.00612;fold3_2017_2019:-0.00275;fold4_2020_2023:-0.00001 |
| station_tseung_kwan_o__w24h_temp_mean_c | 4 | 4723 | 0.925471 | 0.920758 | 0.920166 | -0.0053053 | -0.000592425 | 0.141538 | 4 | 2 | fold1_2011_2013:-0.00698;fold2_2014_2016:0.00676;fold3_2017_2019:-0.00322;fold4_2020_2023:0.00062 |
| station_tuen_mun__w24h_temp_mean_c | 4 | 4732 | 0.927768 | 0.923197 | 0.922608 | -0.00516042 | -0.00058883 | 0.1445 | 4 | 3 | fold1_2011_2013:-0.00577;fold2_2014_2016:0.00620;fold3_2017_2019:-0.00286;fold4_2020_2023:-0.00010 |
| station_ta_kwu_ling__w24h_temp_mean_c | 4 | 4729 | 0.927384 | 0.922698 | 0.922129 | -0.00525531 | -0.000569579 | 0.139422 | 4 | 2 | fold1_2011_2013:-0.00632;fold2_2014_2016:0.00619;fold3_2017_2019:-0.00265;fold4_2020_2023:0.00021 |
| station_tseung_kwan_o__w6h_temp_max_c | 4 | 4705 | 0.925824 | 0.920722 | 0.920198 | -0.00562536 | -0.000523961 | 0.142192 | 4 | 2 | fold1_2011_2013:-0.00676;fold2_2014_2016:0.00517;fold3_2017_2019:-0.00165;fold4_2020_2023:0.00070 |
| station_shek_kong__w24h_max_minus_official_max_c | 4 | 4577 | 0.928709 | 0.924596 | 0.92409 | -0.00461832 | -0.000505623 | 0.163795 | 3 | 2 | fold1_2011_2013:-0.00275;fold2_2014_2016:-0.00449;fold3_2017_2019:0.00281;fold4_2020_2023:0.00190 |
| station_tseung_kwan_o__w12h_temp_mean_c | 4 | 4712 | 0.926125 | 0.921063 | 0.920565 | -0.00555954 | -0.000497478 | 0.141582 | 4 | 2 | fold1_2011_2013:-0.00644;fold2_2014_2016:0.00585;fold3_2017_2019:-0.00236;fold4_2020_2023:0.00058 |
| station_shek_kong__h18_minus_official_max_c | 4 | 4565 | 0.929091 | 0.924874 | 0.924377 | -0.00471413 | -0.000497445 | 0.183296 | 3 | 2 | fold1_2011_2013:-0.00498;fold2_2014_2016:0.00058;fold3_2017_2019:0.00327;fold4_2020_2023:-0.00076 |
| role_inland_nt__w24h_temp_mean_c | 4 | 4744 | 0.92753 | 0.922938 | 0.922466 | -0.00506349 | -0.000472273 | 0.138896 | 4 | 2 | fold1_2011_2013:-0.00621;fold2_2014_2016:0.00628;fold3_2017_2019:-0.00252;fold4_2020_2023:0.00030 |
| station_wong_chuk_hang__w24h_temp_mean_c | 4 | 4711 | 0.927595 | 0.923115 | 0.922653 | -0.00494198 | -0.000461645 | 0.140937 | 4 | 2 | fold1_2011_2013:-0.00658;fold2_2014_2016:0.00645;fold3_2017_2019:-0.00225;fold4_2020_2023:0.00029 |
| station_shek_kong__w6h_max_minus_official_max_c | 4 | 4567 | 0.92875 | 0.924684 | 0.92428 | -0.0044695 | -0.000404051 | 0.185736 | 3 | 2 | fold1_2011_2013:-0.00588;fold2_2014_2016:0.00226;fold3_2017_2019:0.00338;fold4_2020_2023:-0.00116 |
| network__w24h_temp_mean_c | 4 | 4745 | 0.927334 | 0.922767 | 0.922377 | -0.00495717 | -0.000389848 | 0.13924 | 3 | 2 | fold1_2011_2013:-0.00651;fold2_2014_2016:0.00700;fold3_2017_2019:-0.00278;fold4_2020_2023:0.00046 |

## Interpretation

The best signals cluster around official-forecast contradiction and late-window heat state: HKO/network/station temperatures above the official max, 24h maxima, and role/station heat ceilings. That is exactly the mechanism we hoped to see: the official forecast absorbs broad weather level, but it can still lag live thermal evidence.

Humidity/range/overforecast suppression appears in the secondary ranks rather than as a dominant global linear effect. That usually means it should be tested as an interaction with rain/thunderstorm and inland-coastal contrast, not as a standalone linear feature.

No champion changes from this diagnostic run. The next model experiment should promote only the stable top families into a bounded residual or probability specialist, with feature selection frozen inside walk-forward folds.
