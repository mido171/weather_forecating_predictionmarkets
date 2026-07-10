#!/usr/bin/env python3
import re, sys
forbidden=[r'official_underforecast_c',r'official_overforecast_c',r'hot_day_underforecast_flag',r'cold_day_overforecast_flag',r'actual_tmax(?!_training_only)',r'target_tmax_c.*feature']
text=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
hits=[p for p in forbidden if re.search(p,text,re.I)]
print({'file':sys.argv[1],'forbidden_hits':hits})
sys.exit(1 if hits else 0)
