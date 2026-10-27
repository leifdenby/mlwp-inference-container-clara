# CLARA inference artifact

Everything needed to run forecasts with **CLARA** (CRPS-LAM Rapid
Atmospheric ensembles), apart from the input data and the software. CLARA is
the CRPS-LAM model of Larsson, Oskarsson, Landelius and Lindsten (2025),
[arXiv:2510.09484](https://arxiv.org/abs/2510.09484). It was fine-tuned with
a DANRA interior and an IFS forecast boundary.

The artifact is used by the inference container in
[`mlwp-inference-container-clara`](https://github.com/leifdenby/mlwp-inference-container-clara),
which also describes how it was built.

| Path | What |
|---|---|
| `configs/model.yaml` | the model: checkpoint (url, md5), graph, `neural_lam.train_model` arguments, training provenance |
| `configs/danra_model_config_ifs_all.yaml` | the neural-lam config the model was trained with |
| `configs/danra_model1_config_ifs_finetune.yaml` | interior (DANRA) datastore config, PT3H |
| `configs/overlapping_ifs_all2_400km_model1_config.yaml` | boundary (IFS forecast) datastore config: lead times every 6 h, 400 km margin |
| `configs/danra_model1_config.{yaml,zarr}` | the interior statistics datastore: DANRA training statistics |
| `configs/overlapping_era_400km_model1_config.{yaml,zarr}` | the boundary statistics datastore: ERA5 training statistics |
| `grids/danra_model1_config_ifs_finetune.grid.zarr` | the DANRA grid and statics |
| `grids/overlapping_ifs_all2_400km_model1_config.grid.zarr` | the boundary grid (0.25° latitude band of the boundary points, in training order) with the ERA5 statics |
| `graphs/feature_map_graph_198x148/` | the graph the model was trained with |
| `artifact.yaml` | provenance: sources and periods of the statistics, checkpoint md5 |
| `clara_model.ckpt` | the checkpoint: weights, `args` and `config` only. Not part of the published zip; downloaded separately (`clara_inference.fetch_checkpoint`) |

The normalisation statistics are not stored in the checkpoint. They were
recomputed from DANRA v0.5.0 and WeatherBench2 ERA5 over the training
period. If `artifact.yaml` has `placeholder_stats: true`, they are
placeholders and the artifact is for testing only.
