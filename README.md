# audio-diffusion-synth
Comparing diffusion pipelines for synthetic audio generation, as a testbed for distillation on synthetic data.

## Notebooks (run in Colab, in this order)

| Notebook | What it does |
|---|---|
| `notebooks/path_a_mel_diffusion.ipynb` | Path A: 2D UNet on log-mel spectrograms. Creates the shared `frontend.json`, so run it first. |
| `notebooks/path_b_waveform_diffusion.ipynb` | Path B (raw): 1D UNet on mu-law companded waveforms. |
| `notebooks/path_b_codec_diffusion.ipynb` | Path B (token embeddings): 1D UNet on stacked DAC codebook vectors. Kept as a negative result: the model barely learns this representation. |
| `notebooks/path_b_latent_diffusion.ipynb` | Path B (time-domain latent): 1D UNet on DAC's continuous latent reduced with PCA, decoded by DAC. Replaces the token-embedding version. |
| `notebooks/path_c_linear_diffusion.ipynb` | Path C: Path A's model on linear-frequency spectrograms. |
| `notebooks/evaluate_paths.ipynb` | Scores every path against held-out real audio in one space: decoded audio, PANNs and CLAP embeddings, representation ceilings, calibrated Frechet and MMD. |
| `notebooks/tstr_pilot_esc50.ipynb` | Pre-registered TSTR pilot gate: one frozen classifier trained on A-128, B raw and B latent synthetic sets and on matched real data, tested on fold 5. Decides whether the full metric-vs-utility study can run. |
| `notebooks/tstr_sample_sets.ipynb` | Samples the study's new synthetic sets: A-64, A-128 and C at EMA 5k/10k and guidance 1.0/3.0 (9 new sets). Resumable. |
| `notebooks/tstr_study_esc50.ipynb` | The full study: 15 synthetic sets, calibrated Frechet (CLAP, PANNs) against fold 4, TSTR on fold 5 with the pilot's frozen recipe, Spearman correlations with path-bootstrap intervals, within-factor agreement. Resumable. |

## Ocean sounds (grouper / vessel, 10 kHz)

| Notebook | What it does |
|---|---|
| `notebooks/ocean_path_a.ipynb` | Loads and cleans the ocean dataset (duplicates, recording groups, folds), then trains Path A at 128 mel bands and saves samples. |
| `notebooks/ocean_frontend_probe.ipynb` | Tests alternative mel front ends with a small classifier on real clips, a pre-registered decision rule and a 2.5-5 kHz negative control. Result: keep the current front end. |
| `notebooks/ocean_evaluate.ipynb` | Scores the ocean Path A samples against the validation fold: class fidelity under a classifier trained on real clips, calibrated Frechet/MMD overall and per class (in-domain embedder, PANNs, CLAP), ceilings, learnability on validation clips, and a memorization check. |
| `notebooks/ocean_synthetic_only.ipynb` | Trains the probe classifier only on generated ocean clips (Path A, 500 per class) and tests on real fold 5, against the same number of real clips, real clips through Griffin-Lim, and all real clips; classifier front end 10 Hz-1 kHz. |
| `notebooks/ocean_scarce_generators.ipynb` | Retrains ocean Path A on only 64 and on only 256 real clips per class (the scarcity budgets), samples 500 clips per class from each, and checks each for memorisation. |
| `notebooks/ocean_scarcity.ipynb` | Does synthetic data help with few real clips? 16/64/256 real clips per class, real only vs real + synthetic (same-data generator: the honest test; full-data generator: an upper bound), half-real batches, recipe chosen on real only; fold 5. |

Shared code: `common.py` (config, data, front ends, training loop, sampler), `eval_metrics.py` (metrics) and `tstr.py` (the TSTR classifier and its training loop).
