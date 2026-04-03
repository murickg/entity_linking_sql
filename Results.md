```bash
python -m src.run --evaluate

Found 24 instances with gold SQL
[1/24] local003... tables F1=0.6667, cols F1=0.0, time=17.17s
[2/24] local004... tables F1=0.8, cols F1=0.1429, time=21.53s
[3/24] local008... tables F1=1.0, cols F1=0.7368, time=14.37s
[4/24] local017... tables F1=0.6667, cols F1=0.0, time=16.9s
[5/24] local019... tables F1=0.6, cols F1=0.4324, time=14.19s
[6/24] local022... tables F1=1.0, cols F1=0.7317, time=23.41s
[7/24] local023... tables F1=0.8889, cols F1=0.6486, time=22.27s
[8/24] local029... tables F1=1.0, cols F1=0.8182, time=12.12s
[9/24] local038... tables F1=1.0, cols F1=1.0, time=17.74s
[10/24] local039... tables F1=0.0, cols F1=0.0, time=15.43s
[11/24] local058... tables F1=1.0, cols F1=0.7273, time=15.1s
[12/24] local066... tables F1=0.6667, cols F1=0.1333, time=18.39s
[13/24] local065... tables F1=0.6667, cols F1=0.4211, time=21.82s
[14/24] local075... tables F1=0.8, cols F1=0.4, time=18.46s
[15/24] local078... tables F1=1.0, cols F1=0.7143, time=21.55s
[16/24] local099... tables F1=0.8571, cols F1=0.6, time=14.59s
[17/24] local131... tables F1=1.0, cols F1=0.5, time=17.64s
[18/24] local163... tables F1=1.0, cols F1=0.5, time=12.63s
[19/24] local197... tables F1=0.6667, cols F1=0.25, time=10.7s
[20/24] local199... tables F1=1.0, cols F1=0.5333, time=16.12s
[21/24] local210... tables F1=1.0, cols F1=0.7059, time=15.21s
[22/24] local219... tables F1=0.0, cols F1=0.0, time=19.94s
[23/24] local301... tables F1=1.0, cols F1=0.0, time=16.38s
[24/24] local309... tables F1=0.2222, cols F1=0.1379, time=20.18s

============================================================
EVALUATION SUMMARY (24 instances)
============================================================
Tables  - P: 0.7535  R: 0.8338  F1: 0.7709
Columns - P: 0.5762  R: 0.3505  F1: 0.4222
Avg iterations: 6.1, Avg time: 17.2s
```