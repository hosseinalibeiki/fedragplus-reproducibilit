# Evaluation (Reproducible Outputs)

This folder generates the tables and figures used in the Results section from logged experiment artifacts.

## Inputs
- `results/federated_logs.json` produced by `federated/run_federated.py`

## Outputs
- `results/tables/trust_over_rounds.csv`
- `results/figures/fig_trust_over_rounds.png`
- `results/figures/fig_weights_over_rounds.png`

## Run
```bash
python -m eval.make_tables_figures --logs results/federated_logs.json --out_root results
```

## Notes
The evaluation is **label-free** by design, aligned with the paper's scope:
- trust dynamics
- stability and consistency signals
- retrieval score statistics (generated separately via retrieval diagnostics)
