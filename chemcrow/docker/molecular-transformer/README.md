# Native reaction prediction on Linux/AMD64

The published `doncamilom/rxnpred:v3` image is ARM64. Build this directory
locally for an AMD64 machine. This build runs inference on CPU; it does not
require `--gpus all` or ARM emulation.

Run the commands below from the repository root.

## Build

```bash
docker build --platform linux/amd64 -t chemcrow-rxnpred:local \
  chemcrow/docker/molecular-transformer
```

The image pins OpenNMT-py 2.2.0, CPU PyTorch 2.2.2, and NumPy 1.26.4.
Model weights are mounted separately, so they are not copied into the image.
Waitress runs one request thread because the original prediction app uses
shared input/output filenames.

## Download the checkpoint once

Skip this command if `models/USPTO480k_model_step_400000.pt` is already present.
The Google Drive ID is from the original `models/README.md` in this repository.

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  --mount "type=bind,source=$(pwd)/chemcrow/docker/molecular-transformer/models,target=/models" \
  chemcrow-rxnpred:local \
  gdown 1ywJCJHunoPTB5wr6KdZ8aLv7tMFMBHNy \
  -O /models/USPTO480k_model_step_400000.pt
```

## Start

```bash
docker run -d --name chemcrow-rxnpred \
  -p 127.0.0.1:8051:5000 \
  --mount "type=bind,source=$(pwd)/chemcrow/docker/molecular-transformer/models,target=/app/models,readonly" \
  chemcrow-rxnpred:local
```

If the container already exists and is stopped, use `docker start chemcrow-rxnpred`
instead. Inspect it with `docker logs --tail 50 chemcrow-rxnpred`.

## Check inference

```bash
curl --fail-with-body --max-time 120 \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CCO.CC(=O)Cl"}' \
  http://127.0.0.1:8051/api/v1/run
```

Expect a JSON response with `status: "SUCCESS"` and predicted product SMILES.
Successful inference verifies software integration, not chemical accuracy.

## Use from ChemCrow

```bash
conda activate chemcrow
python run_agent.py \
  "Use ReactionPredict to predict the product from reactant SMILES CCO.CC(=O)Cl." \
  --local-rxn --max-iterations 15
```

The runner still uses its configured hosted LLM. `--local-rxn` selects the
local reaction tools; it does not make the entire agent offline. It also
registers retrosynthesis, which needs a separate working service on port 8052.

## Verified on 2026-10-16

- Native `linux/amd64` image built successfully.
- `python -m pip check` passed inside the container.
- Two HTTP prediction requests returned `SUCCESS`.
- The second returned `CCOC(C)=O.Cl` as its first candidate.
- No hosted LLM calls were made during these checks.
