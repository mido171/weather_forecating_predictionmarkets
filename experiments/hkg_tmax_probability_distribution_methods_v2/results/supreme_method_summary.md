# HKG Tmax Probability Engine V2 Supreme Method Summary

Supreme method: `B4_hierarchical_residual_pmf`.
Raw lowest-RPS method: `B5_kernel_analog_pmf`.

The supreme method is chosen by proper scoring rules plus the predeclared V2 promotion gates. A challenger can have an attractive raw score and still fail promotion if its fold 1-4 gain, presealed gain, NLL, Brier, leakage, or row-identity contract does not clear the gate.

## Supreme Row

| rank | method | family | split | row_count | rps | rps_delta_vs_b4 | relative_rps_gain_vs_b4 | nll | brier | crps | ece | mce | entropy | fold14_rps | fold14_relative_rps_gain_vs_b4 | presealed_rps | presealed_relative_rps_gain_vs_b4 | v2_promotion_pass | gates | champion_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | B4_hierarchical_residual_pmf | residual_pmf | fold1_4_plus_presealed_primary | 4747 | 0.041524 | 0.000000 | 0.000000 | 1.037181 | 0.045921 | 0.415236 | 0.019859 | 0.040285 | 1.011110 | 0.041415 | 0.000000 | 0.042121 | 0.000000 | False | reference | True |

## Raw Leaderboard Top 20

| rank | method | family | split | row_count | rps | rps_delta_vs_b4 | relative_rps_gain_vs_b4 | nll | brier | crps | ece | mce | entropy | fold14_rps | fold14_relative_rps_gain_vs_b4 | presealed_rps | presealed_relative_rps_gain_vs_b4 | v2_promotion_pass | gates | champion_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | B5_kernel_analog_pmf | analog_pmf | fold1_4_plus_presealed_primary | 4747 | 0.041287 | -0.000236 | 0.005686 | 1.075467 | 0.045778 | 0.412875 | 0.014419 | 0.044780 | 0.995639 | 0.041194 | 0.005332 | 0.041801 | 0.007601 | False | fail:fold14_rps_gain,presealed_rps_gain,nll | False |
| 2 | H1_b4_challenger_linear_pool | hybrid_pool | fold1_4_plus_presealed_primary | 4747 | 0.041470 | -0.000054 | 0.001290 | 1.034210 | 0.045862 | 0.414700 | 0.016325 | 0.064077 | 1.014058 | 0.041388 | 0.000641 | 0.041919 | 0.004802 | False | fail:fold14_rps_gain,presealed_rps_gain | False |
| 3 | S1_conservative_simplex_stack | stack | fold1_4_plus_presealed_primary | 4747 | 0.041472 | -0.000051 | 0.001232 | 1.032837 | 0.045867 | 0.414724 | 0.018679 | 0.037927 | 1.029417 | 0.041375 | 0.000961 | 0.042007 | 0.002699 | False | fail:fold14_rps_gain,presealed_rps_gain | False |
| 4 | T1_time_decay_b4 | time_decay_residual_pmf | fold1_4_plus_presealed_primary | 4747 | 0.041486 | -0.000038 | 0.000915 | 1.031941 | 0.045941 | 0.414856 | 0.009700 | 0.055938 | 1.031466 | 0.041423 | -0.000188 | 0.041831 | 0.006882 | False | fail:fold14_rps_gain,presealed_rps_gain | False |
| 5 | K2_B4_monotone_cdf_projected | calibration | fold1_4_plus_presealed_primary | 4747 | 0.041524 | 0.000000 | 0.000000 | 1.037181 | 0.045921 | 0.415236 | 0.019859 | 0.040285 | 1.011110 | 0.041415 | 0.000000 | 0.042121 | 0.000000 | False | fail:fold14_rps_gain,presealed_rps_gain,overall_rps_not_better | False |
| 6 | B4_hierarchical_residual_pmf | residual_pmf | fold1_4_plus_presealed_primary | 4747 | 0.041524 | 0.000000 | 0.000000 | 1.037181 | 0.045921 | 0.415236 | 0.019859 | 0.040285 | 1.011110 | 0.041415 | 0.000000 | 0.042121 | 0.000000 | False | reference | True |
| 7 | K0_B4_identity | calibration | fold1_4_plus_presealed_primary | 4747 | 0.041524 | 0.000000 | 0.000000 | 1.037181 | 0.045921 | 0.415236 | 0.019859 | 0.040285 | 1.011110 | 0.041415 | 0.000000 | 0.042121 | 0.000000 | False | fail:fold14_rps_gain,presealed_rps_gain,overall_rps_not_better | False |
| 8 | K1_B4_power_calibrated | calibration | fold1_4_plus_presealed_primary | 4747 | 0.041531 | 0.000007 | -0.000173 | 1.036386 | 0.045904 | 0.415307 | 0.014911 | 0.049582 | 1.031232 | 0.041421 | -0.000147 | 0.042134 | -0.000311 | False | fail:fold14_rps_gain,presealed_rps_gain,overall_rps_not_better | False |
| 9 | B3_forecast_level_residual_pmf | residual_pmf | fold1_4_plus_presealed_primary | 4747 | 0.041616 | 0.000093 | -0.002229 | 1.038655 | 0.045978 | 0.416161 | 0.010817 | 0.127383 | 1.018176 | 0.041548 | -0.003200 | 0.041993 | 0.003027 | False | fail:fold14_rps_gain,presealed_rps_gain,overall_rps_not_better | False |
| 10 | E2_student_t_emos | emos | fold1_4_plus_presealed_primary | 4747 | 0.041658 | 0.000134 | -0.003229 | 1.047414 | 0.046140 | 0.416576 | 0.022522 | 0.110254 | 1.030769 | 0.041584 | -0.004081 | 0.042063 | 0.001380 | False | fail:fold14_rps_gain,presealed_rps_gain,nll,overall_rps_not_better | False |
| 11 | B2_month_residual_pmf | residual_pmf | fold1_4_plus_presealed_primary | 4747 | 0.041666 | 0.000143 | -0.003440 | 1.041036 | 0.046030 | 0.416664 | 0.014703 | 0.115258 | 1.037744 | 0.041648 | -0.005615 | 0.041770 | 0.008332 | False | fail:fold14_rps_gain,presealed_rps_gain,overall_rps_not_better | False |
| 12 | B1_global_residual_pmf | residual_pmf | fold1_4_plus_presealed_primary | 4747 | 0.041700 | 0.000176 | -0.004248 | 1.041529 | 0.046147 | 0.417000 | 0.020253 | 0.086547 | 1.047905 | 0.041687 | -0.006562 | 0.041773 | 0.008271 | False | fail:fold14_rps_gain,presealed_rps_gain,overall_rps_not_better | False |
| 13 | P1_normal_mos | mos | fold1_4_plus_presealed_primary | 4747 | 0.041777 | 0.000254 | -0.006107 | 1.042185 | 0.046279 | 0.417772 | 0.026100 | 0.122924 | 1.057835 | 0.041732 | -0.007656 | 0.042025 | 0.002271 | False | fail:fold14_rps_gain,presealed_rps_gain,nll,overall_rps_not_better | False |
| 14 | P2_student_t_mos | mos | fold1_4_plus_presealed_primary | 4747 | 0.041868 | 0.000345 | -0.008301 | 1.047208 | 0.046332 | 0.418683 | 0.031952 | 0.090748 | 1.106038 | 0.041824 | -0.009883 | 0.042110 | 0.000257 | False | fail:fold14_rps_gain,presealed_rps_gain,nll,overall_rps_not_better | False |
| 15 | E1_normal_emos | emos | fold1_4_plus_presealed_primary | 4747 | 0.042285 | 0.000761 | -0.018334 | 1.097560 | 0.046714 | 0.422849 | 0.048820 | 0.145210 | 0.838563 | 0.042193 | -0.018793 | 0.042789 | -0.015853 | False | fail:fold14_rps_gain,presealed_rps_gain,nll,overall_rps_not_better | False |
| 16 | C1_multinomial_ridge | direct_classifier | fold1_4_plus_presealed_primary | 4747 | 0.042644 | 0.001121 | -0.026987 | 1.056927 | 0.046787 | 0.426442 | 0.013813 | 0.086605 | 1.080459 | 0.042590 | -0.028363 | 0.042944 | -0.019543 | False | fail:fold14_rps_gain,presealed_rps_gain,nll,overall_rps_not_better | False |
| 17 | G1_gamlss_tree_location_scale | gamlss_tree | fold1_4_plus_presealed_primary | 4747 | 0.042795 | 0.001271 | -0.030620 | 1.165812 | 0.047446 | 0.427950 | 0.074955 | 0.158004 | 0.760576 | 0.042689 | -0.030750 | 0.043381 | -0.029912 | False | fail:fold14_rps_gain,presealed_rps_gain,nll,overall_rps_not_better | False |
| 18 | C2_ordinal_cdf_logistic | direct_classifier | fold1_4_plus_presealed_primary | 4747 | 0.043047 | 0.001524 | -0.036694 | 1.064943 | 0.046910 | 0.430472 | 0.017663 | 0.136019 | 1.110760 | 0.042977 | -0.037724 | 0.043432 | -0.031121 | False | fail:fold14_rps_gain,presealed_rps_gain,nll,overall_rps_not_better | False |
| 19 | E3_two_piece_normal_emos | emos | fold1_4_plus_presealed_primary | 4747 | 0.043678 | 0.002154 | -0.051880 | 1.275221 | 0.048587 | 0.436778 | 0.094979 | 0.398077 | 0.698984 | 0.043554 | -0.051653 | 0.044358 | -0.053109 | False | fail:fold14_rps_gain,presealed_rps_gain,nll,brier,overall_rps_not_better | False |
| 20 | Q2_threshold_cdf_gb | threshold_cdf | fold1_4_plus_presealed_primary | 4747 | 0.046938 | 0.005414 | -0.130395 | 1.163617 | 0.047129 | 0.469380 | 0.040806 | 0.099320 | 1.363454 | 0.046746 | -0.128713 | 0.047997 | -0.139495 | False | fail:fold14_rps_gain,presealed_rps_gain,nll,overall_rps_not_better | False |

## Fold 1-4 Scoreboard

| method | family | split | row_count | rps | rps_delta_vs_b4 | relative_rps_gain_vs_b4 | nll | brier | crps | ece | mce | entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B5_kernel_analog_pmf | analog_pmf | fold1_4_primary | 4017 | 0.041194 | -0.000221 | 0.005332 | 1.079702 | 0.045590 | 0.411942 | 0.015956 | 0.075344 | 0.991635 |
| S1_conservative_simplex_stack | stack | fold1_4_primary | 4017 | 0.041375 | -0.000040 | 0.000961 | 1.031156 | 0.045694 | 0.413752 | 0.017095 | 0.072887 | 1.027541 |
| H1_b4_challenger_linear_pool | hybrid_pool | fold1_4_primary | 4017 | 0.041388 | -0.000027 | 0.000641 | 1.033319 | 0.045697 | 0.413885 | 0.015732 | 0.121365 | 1.010134 |
| K2_B4_monotone_cdf_projected | calibration | fold1_4_primary | 4017 | 0.041415 | -0.000000 | 0.000000 | 1.035477 | 0.045742 | 0.414150 | 0.019244 | 0.112094 | 1.007764 |
| K0_B4_identity | calibration | fold1_4_primary | 4017 | 0.041415 | 0.000000 | 0.000000 | 1.035477 | 0.045742 | 0.414150 | 0.019244 | 0.112094 | 1.007764 |
| B4_hierarchical_residual_pmf | residual_pmf | fold1_4_primary | 4017 | 0.041415 | 0.000000 | 0.000000 | 1.035477 | 0.045742 | 0.414150 | 0.019244 | 0.112094 | 1.007764 |
| K1_B4_power_calibrated | calibration | fold1_4_primary | 4017 | 0.041421 | 0.000006 | -0.000147 | 1.034596 | 0.045726 | 0.414211 | 0.013205 | 0.050004 | 1.027773 |
| T1_time_decay_b4 | time_decay_residual_pmf | fold1_4_primary | 4017 | 0.041423 | 0.000008 | -0.000188 | 1.031035 | 0.045794 | 0.414228 | 0.012532 | 0.106292 | 1.030480 |
| B3_forecast_level_residual_pmf | residual_pmf | fold1_4_primary | 4017 | 0.041548 | 0.000133 | -0.003200 | 1.038497 | 0.045815 | 0.415475 | 0.013982 | 0.127383 | 1.013868 |
| E2_student_t_emos | emos | fold1_4_primary | 4017 | 0.041584 | 0.000169 | -0.004081 | 1.045456 | 0.045948 | 0.415840 | 0.023176 | 0.147249 | 1.026588 |
| B2_month_residual_pmf | residual_pmf | fold1_4_primary | 4017 | 0.041648 | 0.000233 | -0.005615 | 1.041724 | 0.045924 | 0.416476 | 0.014388 | 0.154819 | 1.035478 |
| B1_global_residual_pmf | residual_pmf | fold1_4_primary | 4017 | 0.041687 | 0.000272 | -0.006562 | 1.041277 | 0.046036 | 0.416868 | 0.020903 | 0.136488 | 1.046781 |
| P1_normal_mos | mos | fold1_4_primary | 4017 | 0.041732 | 0.000317 | -0.007656 | 1.042242 | 0.046151 | 0.417321 | 0.027643 | 0.122924 | 1.055756 |
| P2_student_t_mos | mos | fold1_4_primary | 4017 | 0.041824 | 0.000409 | -0.009883 | 1.046867 | 0.046206 | 0.418243 | 0.030687 | 0.090748 | 1.103925 |
| E1_normal_emos | emos | fold1_4_primary | 4017 | 0.042193 | 0.000778 | -0.018793 | 1.097529 | 0.046487 | 0.421933 | 0.049288 | 0.188164 | 0.836784 |
| C1_multinomial_ridge | direct_classifier | fold1_4_primary | 4017 | 0.042590 | 0.001175 | -0.028363 | 1.056940 | 0.046598 | 0.425897 | 0.018760 | 0.126128 | 1.076224 |
| G1_gamlss_tree_location_scale | gamlss_tree | fold1_4_primary | 4017 | 0.042689 | 0.001274 | -0.030750 | 1.167115 | 0.047206 | 0.426885 | 0.075591 | 0.158643 | 0.758164 |
| C2_ordinal_cdf_logistic | direct_classifier | fold1_4_primary | 4017 | 0.042977 | 0.001562 | -0.037724 | 1.063884 | 0.046671 | 0.429773 | 0.017050 | 0.145916 | 1.107099 |
| E3_two_piece_normal_emos | emos | fold1_4_primary | 4017 | 0.043554 | 0.002139 | -0.051653 | 1.275831 | 0.048280 | 0.435542 | 0.094974 | 0.398077 | 0.697575 |
| Q2_threshold_cdf_gb | threshold_cdf | fold1_4_primary | 4017 | 0.046746 | 0.005331 | -0.128713 | 1.161120 | 0.046954 | 0.467457 | 0.040772 | 0.143950 | 1.356050 |

## Presealed 2022-2023 Scoreboard

| method | family | split | row_count | rps | rps_delta_vs_b4 | relative_rps_gain_vs_b4 | nll | brier | crps | ece | mce | entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B2_month_residual_pmf | residual_pmf | presealed_2022_2023_primary | 730 | 0.041770 | -0.000351 | 0.008332 | 1.037249 | 0.046610 | 0.417700 | 0.019605 | 0.052738 | 1.050212 |
| B1_global_residual_pmf | residual_pmf | presealed_2022_2023_primary | 730 | 0.041773 | -0.000348 | 0.008271 | 1.042916 | 0.046759 | 0.417725 | 0.020020 | 0.062603 | 1.054092 |
| B5_kernel_analog_pmf | analog_pmf | presealed_2022_2023_primary | 730 | 0.041801 | -0.000320 | 0.007601 | 1.052158 | 0.046812 | 0.418008 | 0.022439 | 0.259017 | 1.017672 |
| T1_time_decay_b4 | time_decay_residual_pmf | presealed_2022_2023_primary | 730 | 0.041831 | -0.000290 | 0.006882 | 1.036922 | 0.046748 | 0.418310 | 0.017779 | 0.225081 | 1.036896 |
| H1_b4_challenger_linear_pool | hybrid_pool | presealed_2022_2023_primary | 730 | 0.041919 | -0.000202 | 0.004802 | 1.039116 | 0.046769 | 0.419187 | 0.026768 | 0.202322 | 1.035653 |
| B3_forecast_level_residual_pmf | residual_pmf | presealed_2022_2023_primary | 730 | 0.041993 | -0.000128 | 0.003027 | 1.039527 | 0.046873 | 0.419934 | 0.012929 | 0.041618 | 1.041882 |
| S1_conservative_simplex_stack | stack | presealed_2022_2023_primary | 730 | 0.042007 | -0.000114 | 0.002699 | 1.042086 | 0.046818 | 0.420072 | 0.033301 | 0.164083 | 1.039740 |
| P1_normal_mos | mos | presealed_2022_2023_primary | 730 | 0.042025 | -0.000096 | 0.002271 | 1.041875 | 0.046982 | 0.420253 | 0.025470 | 0.059043 | 1.069279 |
| E2_student_t_emos | emos | presealed_2022_2023_primary | 730 | 0.042063 | -0.000058 | 0.001380 | 1.058193 | 0.047198 | 0.420628 | 0.023422 | 0.158050 | 1.053774 |
| P2_student_t_mos | mos | presealed_2022_2023_primary | 730 | 0.042110 | -0.000011 | 0.000257 | 1.049088 | 0.047027 | 0.421101 | 0.049880 | 0.303629 | 1.117667 |
| K2_B4_monotone_cdf_projected | calibration | presealed_2022_2023_primary | 730 | 0.042121 | 0.000000 | 0.000000 | 1.046562 | 0.046906 | 0.421209 | 0.032575 | 0.154770 | 1.029523 |
| B4_hierarchical_residual_pmf | residual_pmf | presealed_2022_2023_primary | 730 | 0.042121 | 0.000000 | 0.000000 | 1.046562 | 0.046906 | 0.421209 | 0.032575 | 0.154770 | 1.029523 |
| K0_B4_identity | calibration | presealed_2022_2023_primary | 730 | 0.042121 | 0.000000 | 0.000000 | 1.046562 | 0.046906 | 0.421209 | 0.032575 | 0.154770 | 1.029523 |
| K1_B4_power_calibrated | calibration | presealed_2022_2023_primary | 730 | 0.042134 | 0.000013 | -0.000311 | 1.046234 | 0.046882 | 0.421340 | 0.028671 | 0.135221 | 1.050268 |
| E1_normal_emos | emos | presealed_2022_2023_primary | 730 | 0.042789 | 0.000668 | -0.015853 | 1.097734 | 0.047966 | 0.427886 | 0.047737 | 0.499093 | 0.848352 |
| C1_multinomial_ridge | direct_classifier | presealed_2022_2023_primary | 730 | 0.042944 | 0.000823 | -0.019543 | 1.056852 | 0.047827 | 0.429441 | 0.023454 | 0.334220 | 1.103761 |
| G1_gamlss_tree_location_scale | gamlss_tree | presealed_2022_2023_primary | 730 | 0.043381 | 0.001260 | -0.029912 | 1.158647 | 0.048769 | 0.433808 | 0.072151 | 0.191339 | 0.773848 |
| C2_ordinal_cdf_logistic | direct_classifier | presealed_2022_2023_primary | 730 | 0.043432 | 0.001311 | -0.031121 | 1.070774 | 0.048223 | 0.434318 | 0.035385 | 0.139785 | 1.130907 |
| E3_two_piece_normal_emos | emos | presealed_2022_2023_primary | 730 | 0.044358 | 0.002237 | -0.053109 | 1.271868 | 0.050278 | 0.443579 | 0.095588 | 0.212130 | 0.706733 |
| Q2_threshold_cdf_gb | threshold_cdf | presealed_2022_2023_primary | 730 | 0.047997 | 0.005876 | -0.139495 | 1.177363 | 0.048088 | 0.479966 | 0.048591 | 0.346618 | 1.404201 |

Interpretation rule: B4 remains the default champion unless a challenger clears all promotion gates. This prevents choosing a more complex probability engine from a marginal, unstable, or poorly calibrated score difference.