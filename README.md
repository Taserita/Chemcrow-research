# ChemCrow research pipeline

ChemCrow combines an LLM agent with chemistry tools and literature retrieval.
This research checkout adds local PDF and Semantic Scholar research, structured
evidence with source provenance, and evidence-grounded solution design to the
original ChemCrow toolset.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/chemcrow_dark_bold.png">
  <img alt="ChemCrow logo" src="assets/chemcrow_light_bold.png" width="100%">
</picture>

Features include molecular conversion and analysis, controlled-chemical and
explosive checks, optional reaction prediction and retrosynthesis, and saved
research evidence, solution plans, and actual agent observations.

## Pipeline and implementation status

| Stage | Implementation | Current connection |
| --- | --- | --- |
| Literature review | `ScientificResearch`, local PDFs, Semantic Scholar, PaperQA | Registered in the agent; model selects when to call it |
| Evidence extraction | Findings and limitations with program-assigned source metadata | Runs within `ScientificResearch`; saves `evidence.json` |
| Solution design | `SolutionDesign` and strict Pydantic schemas | Registered; consumes saved evidence and creates a pending evaluation plan |
| Tool evaluation | Existing chemistry tools; independent `evaluation.evaluate_plan()` controller | Agent executes tools by model choice; controller is **not wired into the agent or CLI** |
| Final assessment | Agent final response; separate controller assessment/report | Agent can finish early; completing every planned check is not enforced |

The agent prompts describe the full scientific workflow and retain the existing
chemical safety rules. With `--research`, only literature research and solution
design are available; chemistry checks are deferred. A generated plan does not establish that checks ran or
that a solution was validated. The standalone controller is preserved without
adding new orchestration. See [research workflow and JSON contracts](docs/research.md).

## Installation

Install this checkout from the repository root; the upstream PyPI package does
not include these local research changes. Python 3.9–3.11 is declared supported;
this cleanup was validated with Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r dev-requirements.txt  # for tests and development
```

Git is required to install the pinned `paper-scraper` revision. `setup.py` is the
single runtime dependency definition. It preserves OpenAI 0.27.8 and PaperQA
1.1.1, pins the tested LangChain 0.0.275 / rmrkl 0.0.3 stack, and declares directly
imported dependencies. This is a legacy stack; do not upgrade its major components
independently. The supplied dependency configuration is not a complete lockfile.

## Configuration

Copy `.env.example` to `.env` only if `.env` does not already exist, then fill in
the services you use. `run_agent.py` loads `.env` beside the script. Shell variables
take precedence. No credentials are included in the example.

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required for the CLI agent, hosted helper LLM, and PDF embeddings |
| `SEMANTIC_SCHOLAR_API_KEY` | Online paper discovery in combined research mode |
| `SERP_API_KEY` | Optional SerpAPI web discovery and `WebSearch` tool |
| `RXN4CHEM_API_KEY` | Optional remote reaction tools; requires a project ID too |
| `RXN4CHEM_PROJECT_ID` | Your RXN project ID; no built-in project is used |
| `CHEMSPACE_API_KEY` | Optional molecule prices and structure lookup fallback |
| `RESEARCH_SOURCE_MODE` | `local` or `combined`; CLI and tool registry default to `local` |
| `RESEARCH_PDF_DIR` | Optional nonrecursive directory of local PDFs |

Leave unused variables unset; uncomment research settings only when supplying
values. No key is needed for PubChem, Wikipedia, or
molbloom catalog access, but these tools can still use the network or download data.

## Run the agent

```bash
python run_agent.py --help
python run_agent.py "What is the exact molecular weight of ethanol?" \
  --source-mode combined --model gpt-4.1-mini --max-iterations 10
```

The CLI default is `gpt-4.1-mini`. Choose a model supported by the legacy client
and available to your account. Python callers can use `ChemCrow` directly:

```python
from chemcrow.agents import ChemCrow

agent = ChemCrow(model="gpt-4.1-mini", tools_model="gpt-4.1-mini", streaming=False)
answer = agent.run("What is the exact molecular weight of ethanol?")
```

## Research with local PDFs

Place papers you are authorized to process in `local_papers/` (ignored by Git).
PDFs need extractable text and a lowercase `.pdf` extension; OCR is not included.

```bash
python run_agent.py \
  "Review the local papers with ScientificResearch, use SolutionDesign to propose a supported solution and evaluation plan, execute available checks, and report actual observations and limitations." \
  --source-mode local --pdf-dir ./local_papers \
  --max-passages 6 --max-evaluation-steps 12 --max-iterations 20
```

For literature research and solution design **without executing chemistry tools**:

```bash
python run_agent.py "Review this study and design an evidence-grounded solution and evaluation plan." \
  --research --pdf ./local_papers/study.pdf --max-passages 4 --max-iterations 20
```

Research mode exposes only `ScientificResearch` and `SolutionDesign` to the agent.
Its planner can describe later chemistry checks, but those checks are not executed
in this run. The model still chooses its tool calls; completing both stages is not
guaranteed. Retrieval alone remains available through the Python
`local_literature_search()` API.

Source selection follows `--source-mode`, then `RESEARCH_SOURCE_MODE`, then `local`.
Local mode requires `--pdf`, `--pdf-dir`, or `RESEARCH_PDF_DIR`.
Repeat `--pdf` to select multiple files. Local source mode skips online paper
search, but **still sends document text to hosted embeddings and LLMs**. Other
agent tools retain their normal network behavior. Use `--source-mode combined`
to add online discovery; local papers remain usable if that discovery fails.

## Reaction services

`--local-rxn` registers both local services: prediction on port 8051 and
retrosynthesis on port 8052. Configure and start them separately using the
[service documentation](chemcrow/docker/README.md). Registration does not check
service health. Without this flag, remote reaction tools are registered only
when both `RXN4CHEM_API_KEY` and `RXN4CHEM_PROJECT_ID` are configured.
In research mode, reaction capabilities appear only in the deferred evaluation
catalog. The remote summary uses returned route information without adding
unspecified solvents or conditions.

## Saved results

| Location | Contents |
| --- | --- |
| `research_outputs/<run>/` | Discovery, original passages, evidence, extraction audit, stage trace; downloaded PDFs and selection/screening audits in combined mode |
| `research_outputs/solution_<run>/` | Solution plan, evidence snapshot, tool catalog, generation audit, trace |
| `runs/<timestamp>.json` | Agent tool inputs, actual observations, model output, final answer, call counts |
| `<solution-dir>/evaluation_<run>/` | Independent controller output, only when `evaluate_plan()` is called explicitly |

`--trace new-file.json` overrides the CLI trace location and refuses to overwrite
an existing file. Agent traces include `research_only` and `source_mode`.
Evidence retains its six-field summary format and older evidence
files remain readable. Artifact schemas and paths are detailed in
[docs/research.md](docs/research.md).

Outputs can contain research text, questions, absolute paths and `file://` URIs.
Keep them private unless separately reviewed and sanitized. `.gitignore` excludes
these outputs, credentials, local PDFs, caches, environments, and model weights.

## Validation

```bash
python -m pytest -q
python run_agent.py --help
```

Only the five original ChemCrow test files are retained. Default pytest settings
exclude tests marked `integration`; there is no repository-wide network blocker.
The removed research regression and temporary validation tests are not part of CI.
To intentionally run network-dependent tests with configured services:

```bash
python -m pytest -o addopts='' -m integration
```

Integration credentials must be supplied through the environment; tests do not
load `.env`. Existing explicitly skipped hosted-LLM tests remain skipped.
See [the cleanup and release report](RELEASE_REVIEW.md) for actual validation results.

## Limitations and release review

- Research, planning, and final interpretation depend on model output. Provenance
  validation checks source IDs and page ranges, not scientific entailment.
- The main agent may stop before every planned check. Only explicitly invoking the
  separate POSIX controller provides its execution accounting and bounded retries.
- Predictions do not establish experimental success, yield, purity, or safety.
  The original safety checks are preserved and are not a comprehensive hazard assessment.
- Online papers can be inaccessible or lack usable text; relevance screening is
  model-based and retrieval budgets limit coverage.
- Legacy helpers include broad exception handling and some unbounded HTTP calls.
  The standalone evaluation controller does not change the main agent's timeouts.
- Live APIs and Docker services were not exercised during cleanup. Fresh installation
  on all supported Python versions still needs validation.
- This checkout does not contain every tool from the original paper and does not
  reproduce all of its published results. Original experiments are available in
  [ChemCrow runs](https://github.com/ur-whitelab/chemcrow-runs).

The upstream PyPI publishing workflow is archived as
`.github/workflows/build.yml.disabled`, so creating a GitHub release does not
activate it. Publishing this fork to PyPI would require a separate package name
and dependency review.

## Attribution
Bran, Andres M., et al. "ChemCrow: Augmenting large-language models with chemistry tools." arXiv preprint arXiv:2304.05376 (2023).

```bibtex
@article{bran2023chemcrow,
      title={ChemCrow: Augmenting large-language models with chemistry tools},
      author={Andres M Bran and Sam Cox and Oliver Schilter and Carlo Baldassari and Andrew D White and Philippe Schwaller},
      year={2023},
      eprint={2304.05376},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      publisher={arXiv}
}
```

The original [MIT license](LICENSE), author metadata, and bundled data attribution
are retained. Review third-party dataset and model redistribution terms separately.
