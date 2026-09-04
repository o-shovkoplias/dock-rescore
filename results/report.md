# dock-rescore report

Complexes: 5, poses: 89, features: 70
CV: GroupKFold n_folds=5 over 5 complexes

| metric | Vina rank | ML rescoring |
|---|---|---|
| top1_success | 0.800 | 0.800 |
| top1_hits | 4 | 4 |
| top3_near_native_fraction | 0.333 | 0.333 |
| top3_enrichment | 5.933 | 5.933 |
| roc_auc | 0.857 | 0.721 |

Oracle top-1 (any near-native pose): 0.800
Near-native pose fraction: 0.056

## Timing per complex (s)

| complex_id   |   prepare |   dock |   rmsd |   features |   total_s |
|:-------------|----------:|-------:|-------:|-----------:|----------:|
| 7D5C_GV6     |       1.0 |   12.0 |    0.0 |        5.2 |      18.2 |
| 7N03_ZRP     |       0.0 |    6.9 |    0.0 |        2.5 |       9.4 |
| 7P1M_4IU     |       0.1 |    4.5 |    0.0 |        2.0 |       6.6 |
| 7TUO_KL9     |       0.1 |    5.8 |    0.0 |        3.6 |       9.5 |
| 7YZU_DO7     |       0.1 |    2.6 |    0.0 |        2.8 |       5.5 |

Mean total per complex: 9.8 s

## Software versions

- python: 3.11.16
- platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.39
- rdkit: 2026.03.1
- vina: 1.2.7
- meeko: 0.8.0
- openbabel: 3.2.1
- prolif: 2.2.1
- MDAnalysis: 2.10.0
- torch: 2.13.0
- sklearn: 1.9.0
- numpy: 2.4.6
- pandas: 3.0.5
