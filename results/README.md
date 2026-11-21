# Results

Every file here is written by a stage in `src/`. The tables are the reference outputs behind the paper;
`scripts/reproduce_stats.sh` regenerates those written by `s09`, `s12` and `s11` (Figs 2-3) from `preds/` and
compares them with the shipped versions. The paper-item map is in the top-level `README.md`.

The `regime` column of every table uses the paper names (T-orig, T-jitter, T-paste, T-shuffle, T-rand); the
`T-paste-U2Net (suppl.)` and `T-rand-U2Net (suppl.)` rows are the supplementary four-class suite.

## Per-image predictions (`preds/`)

| File name | Writer | Contents |
|---|---|---|
| `<arch>_<regime>_s<seed>.npz` | `s08_cnn` | Softmax probabilities, shape (n, 6), per evaluation set |
| `svm_<feat>_<regime>_d<draw>.npz` | `s07_svm` | Predicted class indices (n,) per set, plus `rw_dec`, the RealWaste decision values (n, 6) |
| `m0_orig_d0.npz` | `s07_svm` | Majority-class predictions |

`arch`: `cnn` (partly frozen ResNet-18, the paper's primary model), `resnet18_ft`, `mobilenet_v3`, `swin_t`.
`feat`: `hist` (colour SVM), `sift` (SIFT SVM). `seed` 0-2 for networks; `draw` 0-2 for the random paste draws of
augmented SVMs (0 only for un-augmented regimes and probes).

`regime`: `orig`, `jitter`, `paste6`, `shuffle6`, `rand6` (paper: T-orig, T-jitter, T-paste, T-shuffle, T-rand);
`paste`, `rand` (four-class U2-Net suite, supplementary); background-only probes `probe_bg6` and `probe_corners`
(paper), `probe_bg` and `probe_fixed` (four-class, supplementary). Probe files hold only the matching probe set.
Network probe files are keyed by set name (e.g. `var6_base_ONLY_BG_T`, `probe_corners`). SVM probe files use the
keys `probe`, `probe_erode3` and `probe_dilate3`: for `probe_bg6` and `probe_bg` these are the base, erode3 and
dilate3 masks of `ONLY_BG_T` (`var6_*` and `var_*` respectively); `probe_corners` and `probe_fixed` hold only
`probe`.

Keys (evaluation sets), in the order of `src/testsets.py`:

| Key | Set | n |
|---|---|---|
| `tn_val`, `tn_test` | TrashNet validation and test split (`data/work/split.csv`) | 377, 378 |
| `rw` | RealWaste, six shared classes | 3,587 |
| `var6_<masks>_<variant>` | Six-class edited test images from human polygons | 258 |
| `var_<masks>_<variant>` | Four-class edited test images from U2-Net masks (supplementary) | 152 |
| `probe_corners`, `probe_fixed` | Corner probe (16 × 16 corners); fixed-box probe (supplementary) | 194, 35 |

`masks`: `base`, `erode3`, `dilate3`. `variant`: `ORIGINAL`, `ONLY_FG`, `ONLY_BG_T`, `MIXED_SAME_<p>`,
`MIXED_RAND_<p>` with pairing `p` = 0-4 (pairings in `data/work/variants6/test_pairings.json`). Row order matches
the metadata tables in `data/work/variants6/` and `data/work/variants/`.

## Tables

| File | Writer | Contents |
|---|---|---|
| `counts.csv`, `split_counts.csv` | `s01`, `s03` | Images per class in each dataset and split subset |
| `duplicate_audit.csv` | `s02` | Duplicate audit and agreement with Prayitno & Putra (2026) |
| `mask_quality.csv`, `mask_gate.json`, `crop_loss_reference.csv`, `eligibility_counts.csv` | `s04` | U2-Net and Otsu masks against the reference polygons, method gate, crop loss, four-class eligibility |
| `variants_info.json`, `variants6_info.json`, `eligibility6_counts.csv` | `s05_variants`, `s05_variants6` | Sizes of the edited sets and probes |
| `svm_tuning.csv`, `m0.json` | `s07` | Chosen C and γ per feature, regime and draw; majority class |
| `nn_runs.csv`, `logs/*.json` | `s08` | One row per shipped network run: `s08_version` (v1 for the twelve four-class `cnn` runs `paste`, `rand`, `probe_bg`, `probe_fixed`; v2 for all others; see "Two revisions of `s08_cnn.py`" in the top-level README), epochs, best epoch, minutes, device, GPU, torch version; per-epoch curves (`val_macro_f1`, `last_batch_loss`, `sec`) |
| `rq4.csv`, `rq4.json` | `s10` | Accuracy on official-split test images with and without a training duplicate |
| `performance.csv`, `performance_seed_stats.csv`, `confusion_matrices.npy` | `s09` | Accuracy, macro-F1, balanced accuracy and per-class scores with 95% CIs for every model, regime, seed and ensemble (`rep` = seed or `ens`); seed mean and SD; confusion matrices (a pickled dict, load with `allow_pickle=True`) |
| `probes.csv` | `s09` | Background-only probes: balanced accuracy, CI and permutation p-value |
| `reliance6.csv`, `reliance6_per_class.csv` | `s09` | Six-class edited images: accuracy per variant, BG-GAP, flip rate, chance-corrected gap |
| `reliance.csv`, `reliance_per_class.csv` | `s09` | The same for the four-class suite (supplementary) |
| `mask_quality_covariate.csv`, `error_modes.csv` | `s09` | Flip rate against mask IoU; RealWaste error types |
| `summary.json` | `s09` | Planned tests (H1, H2, H3) and secondary tests with Holm-adjusted p-values; keys below |
| `regime_contrasts.csv` | `s12` | RealWaste regime contrasts per backbone (families A and B, Holm) |
| `seed_variability.csv` | `s12` | Ensemble values with CIs next to single-seed mean, SD and range |
| `exclusion_reasons.csv`, `bggap_sensitivity.csv` | `s12` | Reasons for exclusion from the edited set; BG-GAP on stricter subsets and class-balanced |
| `sorting_errors.csv`, `error_cost.csv` | `s12` | Recyclable-to-trash and trash-to-recycling rates; error-cost sensitivity |
| `contrasts.json` | `s12` | All `s12` results in one file, including the RealWaste duplicate summary |
| `figures/icasf_fig{1,2,3}_*.png` | `s11` | Paper Figures 1-3 |

## Keys of `summary.json`

| Key | Contents |
|---|---|
| `planned_tests` | H1, H2, H3 for the partly frozen ResNet-18 (six-class suite): raw `p` and Holm-adjusted `p_holm` over the three; H3 also `seeds_agree_positive`. H1 is the larger permutation p-value of the two probes (`bg6`, `corners`); H2 the sign-flip test of BG-GAP; H3 the one-sided paired bootstrap of RealWaste macro-F1, T-rand − T-paste |
| `H3_six_class`, `H3_five_class` | Details of H3 on RealWaste with six classes and with trash dropped: macro-F1 per regime (ensemble and seeds), differences T-rand − T-paste, T-rand − T-orig, T-paste − T-orig, T-jitter − T-orig with 95% CIs, McNemar counts |
| `gap_closed` | Share of the TrashNet-to-RealWaste macro-F1 gap recovered by T-rand, relative to T-paste and to T-orig |
| `multi_arch` | `realwaste_6class`, `realwaste_5class` and `gap_closed_6class` (as above) for each network |
| `secondary` | Secondary family with Holm-adjusted p-values: chance-corrected BG-GAP comparisons (`RQ2 ...`), H1 and H2 analogues for the SVMs, McNemar tests, duplicate-leakage results (`RQ4 ...`, from `rq4.json`) |
| `supplementary_u2net` | The same planned tests and H3 details on the four-class U2-Net suite (not in the paper) |
