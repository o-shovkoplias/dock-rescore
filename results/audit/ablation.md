# Feature-ablation audit (GroupKFold(5) by complex, out-of-fold, 273 complexes / 4745 poses)

Mean over MLP seeds 0,1,2 (min-max in brackets). `pooled AUC` is per-pose over all complexes
(inflated by easy-vs-hard complex differences); `per-complex AUC` is the mean AUC within each
complex that has both classes, which is what re-ranking actually needs.

| variant | n_feat | top-1 success | pooled ROC-AUC | per-complex AUC | top-3 enrichment |
|---|---|---|---|---|---|
| Vina rank | 1 | 0.722 | 0.694 | 0.935 | 4.579 |
| centroid offset alone (no model) | 1 | 0.744 | 0.966 | 0.966 | 4.938 |
| full (as reported) | 70 | 0.791 [0.788-0.795] | 0.949 [0.949-0.950] | 0.971 [0.970-0.972] | 5.063 [5.058-5.073] |
| drop geo_centroid_offset | 69 | 0.716 [0.711-0.718] | 0.905 [0.903-0.907] | 0.935 [0.932-0.938] | 4.579 [4.549-4.594] |
| drop geo_centroid_offset + cons_* | 66 | 0.719 [0.707-0.729] | 0.880 [0.877-0.885] | 0.929 [0.927-0.931] | 4.629 [4.594-4.684] |
| drop geo_centroid_offset + cons_* + plf_* | 55 | 0.700 [0.696-0.707] | 0.874 [0.871-0.877] | 0.925 [0.923-0.926] | 4.485 [4.445-4.534] |
| Vina terms only | 8 | 0.719 [0.718-0.722] | 0.896 [0.893-0.899] | 0.933 [0.928-0.936] | 4.594 [4.579-4.624] |

95% paired-bootstrap CI (over complexes) of top-1(ML) - top-1(Vina), seed 0:
- full (as reported): [+0.033, +0.103]
- drop geo_centroid_offset: [-0.033, +0.026]
- drop geo_centroid_offset + cons_*: [-0.029, +0.044]
- drop geo_centroid_offset + cons_* + plf_*: [-0.055, +0.004]
- Vina terms only: [-0.015, +0.007]
