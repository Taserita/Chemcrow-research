# Optional local reaction services

The agent selects these clients with `--local-rxn`. Services must be configured
and started separately; the agent never starts Docker.

| Tool | Default endpoint | Setup |
| --- | --- | --- |
| `ReactionPredict` | `http://localhost:8051/api/v1/run` | [Molecular Transformer](molecular-transformer/README.md) |
| `ReactionRetrosynthesis` | `http://localhost:8052/api/v1/retrosynth` | [AiZynthFinder](aizynthfinder/README.md) |

Model weights, stock data and generated input/output files are not included in the
Python package. Preserve their upstream licenses when obtaining them.

The source AiZynthFinder app in this tree exposes `/api/v1/run`, whereas the
published image described in its README uses `/api/v1/retrosynth`. The client
supports an explicit `endpoint` override for the older source app. Both accept a
`target` field. Prediction accepts a `smiles` field.

The service READMEs retain historical verification notes; no service was started
or retested during release cleanup. Tool registration is not a service health check.
