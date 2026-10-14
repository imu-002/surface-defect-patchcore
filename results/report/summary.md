
### Standard: detection, localisation and speed per category

| Category | Image AUROC | Pixel AUROC | AU-PRO | F1 @ calibrated thr. | FPR @ thr. | Recall @ thr. | Best F1 (oracle) | ms/image | FPS |
|---|---|---|---|---|---|---|---|---|---|
| brick | 0.998 | 0.988 | 0.969 | 0.930 | 0.300 | 1.000 | 0.988 | 350 | 2.9 |
| fabric | 1.000 | 0.993 | 0.980 | 0.982 | 0.075 | 1.000 | 1.000 | 356 | 2.8 |
| grass | 0.858 | 0.977 | 0.869 | 0.832 | 0.775 | 0.988 | 0.846 | 335 | 3.0 |
| gravel | 0.924 | 0.985 | 0.903 | 0.779 | 0.000 | 0.637 | 0.902 | 330 | 3.0 |
| leather | 0.989 | 0.980 | 0.950 | 0.974 | 0.000 | 0.950 | 0.974 | 320 | 3.1 |
| wood | 0.990 | 0.972 | 0.941 | 0.947 | 0.000 | 0.900 | 0.975 | 345 | 2.9 |
| **Mean** | **0.960** | **0.983** | **0.935** | **0.907** | **0.192** | **0.913** | **0.948** | **339** | **2.9** |

**Standard: image AUROC by defect type (good vs that type)**

| Category | color | crack | hole | scratch | stain |
|---|---|---|---|---|---|
| brick | 1.000 | 1.000 | 1.000 | 1.000 | 0.992 |
| fabric | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| grass | 0.925 | 0.781 | 0.836 | 0.816 | 0.930 |
| gravel | 1.000 | 0.883 | 0.736 | 1.000 | 1.000 |
| leather | 0.986 | 1.000 | 1.000 | 1.000 | 0.959 |
| wood | 0.994 | 1.000 | 1.000 | 1.000 | 0.955 |

### Subtle: detection, localisation and speed per category

| Category | Image AUROC | Pixel AUROC | AU-PRO | F1 @ calibrated thr. | FPR @ thr. | Recall @ thr. | Best F1 (oracle) | ms/image | FPS |
|---|---|---|---|---|---|---|---|---|---|
| brick | 0.867 | 0.915 | 0.842 | 0.828 | 0.300 | 0.812 | 0.840 | 348 | 2.9 |
| fabric | 0.926 | 0.959 | 0.900 | 0.806 | 0.075 | 0.700 | 0.901 | 353 | 2.8 |
| grass | 0.490 | 0.608 | 0.223 | 0.679 | 0.775 | 0.713 | 0.794 | 357 | 2.8 |
| gravel | 0.610 | 0.780 | 0.413 | 0.202 | 0.000 | 0.113 | 0.804 | 334 | 3.0 |
| leather | 0.703 | 0.729 | 0.476 | 0.416 | 0.000 | 0.263 | 0.814 | 361 | 2.8 |
| wood | 0.756 | 0.744 | 0.623 | 0.584 | 0.000 | 0.412 | 0.804 | 326 | 3.1 |
| **Mean** | **0.725** | **0.789** | **0.580** | **0.586** | **0.192** | **0.502** | **0.826** | **347** | **2.9** |

**Subtle: image AUROC by defect type (good vs that type)**

| Category | color | crack | hole | scratch | stain |
|---|---|---|---|---|---|
| brick | 0.775 | 1.000 | 1.000 | 0.934 | 0.623 |
| fabric | 0.980 | 0.986 | 0.852 | 1.000 | 0.814 |
| grass | 0.475 | 0.353 | 0.523 | 0.566 | 0.534 |
| gravel | 0.961 | 0.381 | 0.566 | 0.609 | 0.533 |
| leather | 0.745 | 0.637 | 0.519 | 1.000 | 0.616 |
| wood | 0.514 | 0.980 | 0.791 | 0.984 | 0.509 |

### Image-level AUROC: PatchCore vs unsupervised global-feature baselines (mean over categories)

| Dataset | PatchCore (ours) | Global-feature kNN | One-class SVM | Isolation Forest |
|---|---|---|---|---|
| Standard | 0.960 | 0.881 | 0.866 | 0.911 |
| Subtle | 0.725 | 0.659 | 0.667 | 0.692 |

### Supervised classifiers vs PatchCore when the defect type was NOT in the training data (mean image AUROC)

| Dataset | Scenario | SVM (supervised) | Random Forest (supervised) | PatchCore (no defect labels) |
|---|---|---|---|---|
| Standard | seen defect types | 0.964 | 0.956 | 0.965 |
| Standard | unseen defect types | 0.948 | 0.914 | 0.957 |
| Subtle | seen defect types | 0.754 | 0.736 | 0.704 |
| Subtle | unseen defect types | 0.734 | 0.735 | 0.716 |

### Defect-type identification: 5-fold cross-validated accuracy on defective images (mean over categories)

| Dataset | Features | kNN | SVM | Random Forest |
|---|---|---|---|---|
| Standard | shape+colour only | 0.823 | 0.854 | 0.850 |
| Standard | deep features only | 0.896 | 0.954 | 0.944 |
| Standard | both | 0.898 | 0.958 | 0.931 |
| Subtle | shape+colour only | 0.508 | 0.498 | 0.567 |
| Subtle | deep features only | 0.502 | 0.548 | 0.569 |
| Subtle | both | 0.502 | 0.546 | 0.585 |

### Ablation (ablation_standard): embedding size x memory-bank size

| Category | Embedding dim | Coreset ratio | Bank patches | Image AUROC | Pixel AUROC | AU-PRO | Search ms | Est. FPS |
|---|---|---|---|---|---|---|---|---|
| leather | 1536 | 0.01 | 870 | 0.993 | 0.981 | 0.952 | 16 | 7.5 |
| leather | 1536 | 0.05 | 4352 | 0.991 | 0.980 | 0.950 | 71 | 5.3 |
| leather | 1536 | 0.1 | 8704 | 0.989 | 0.980 | 0.950 | 150 | 3.7 |
| leather | 128 | 0.01 | 870 | 0.972 | 0.974 | 0.937 | 4 | 8.2 |
| leather | 128 | 0.05 | 4352 | 0.974 | 0.978 | 0.947 | 13 | 7.6 |
| leather | 128 | 0.1 | 8704 | 0.973 | 0.977 | 0.946 | 36 | 6.5 |
| wood | 1536 | 0.01 | 870 | 0.988 | 0.970 | 0.939 | 18 | 6.7 |
| wood | 1536 | 0.05 | 4352 | 0.988 | 0.972 | 0.941 | 67 | 5.0 |
| wood | 1536 | 0.1 | 8704 | 0.990 | 0.972 | 0.941 | 183 | 3.2 |
| wood | 128 | 0.01 | 870 | 0.923 | 0.946 | 0.900 | 3 | 7.4 |
| wood | 128 | 0.05 | 4352 | 0.939 | 0.959 | 0.913 | 13 | 6.9 |
| wood | 128 | 0.1 | 8704 | 0.947 | 0.960 | 0.910 | 41 | 5.8 |

### Ablation (ablation_subtle): embedding size x memory-bank size

| Category | Embedding dim | Coreset ratio | Bank patches | Image AUROC | Pixel AUROC | AU-PRO | Search ms | Est. FPS |
|---|---|---|---|---|---|---|---|---|
| fabric | 1536 | 0.01 | 870 | 0.921 | 0.954 | 0.894 | 18 | 7.7 |
| fabric | 1536 | 0.1 | 8704 | 0.926 | 0.959 | 0.900 | 139 | 4.0 |
| wood | 1536 | 0.01 | 870 | 0.725 | 0.719 | 0.620 | 18 | 8.1 |
| wood | 1536 | 0.1 | 8704 | 0.756 | 0.744 | 0.623 | 144 | 4.0 |
