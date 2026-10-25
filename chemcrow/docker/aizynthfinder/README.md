# Retrosynthesis service

The verified public image is `doncamilom/retrosynth:v3` (Linux/AMD64,
AiZynthFinder 4.3.0). Its API differs from the older source app in this directory.

```bash
docker run -d --name chemcrow-retrosynth \
  -p 127.0.0.1:8052:5000 doncamilom/retrosynth:v3
```

If the container already exists and is stopped, use
`docker start chemcrow-retrosynth`. Inspect it with
`docker logs --tail 50 chemcrow-retrosynth`.

## Test the published image directly

```bash
curl --fail-with-body --max-time 240 \
  -H 'Content-Type: application/json' \
  -d '{"target":"CC(=O)Oc1ccccc1C(=O)O"}' \
  http://127.0.0.1:8052/api/v1/retrosynth
```

The response is a JSON list of route trees. The published image expects
`target`, not `smiles`, and `/api/v1/retrosynth`, not `/api/v1/run`.
Submit requests sequentially: this image uses a shared `trees.json` file.

## Use ChemCrow

From the repository root:

```bash
conda activate chemcrow
python run_agent.py \
  "Use ReactionRetrosynthesis to propose a route for aspirin, SMILES CC(=O)Oc1ccccc1C(=O)O. Report only what the tool provides." \
  --local-rxn --max-iterations 15
```

`RXNRetrosynthesisLocal` now returns the first route's target, solved status,
and mapped reaction steps as JSON, in forward synthesis order. The main
agent can explain this evidence. The local tool itself makes no OpenAI calls
and does not invent solvents, quantities, conditions, or yields.
Other returned routes remain accessible through `tool.retrosynthesis(smiles)`.

If building the older source app in this directory, its endpoint remains
`/api/v1/run`. Configure the wrapper explicitly for that service:

```python
tool = RXNRetrosynthesisLocal(endpoint="http://localhost:8052/api/v1/run")
```

## Verification on 2026-10-16

- The old client URL returned HTTP 404.
- The published image's actual endpoint returned HTTP 200 and 12 routes for
  aspirin in approximately 14 seconds; all were marked solved by the planner.
- All 12 routes could be parsed by the repaired wrapper.
- A fresh full wrapper call succeeded in approximately 13 seconds.
- Ten offline regression tests passed in `tests/test_local_retrosynthesis.py`.

These checks establish integration and route parsing, not experimental validity
or the accuracy of every proposed route. The hosted agent was not invoked.
