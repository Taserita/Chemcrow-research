# ChemCrow research agent

ChemCrow connects literature research, evidence extraction, solution design, and
chemistry tools in one LLM agent. This repository extends the original ChemCrow
with local PDF retrieval, Semantic Scholar discovery, and saved research evidence
and evaluation plans.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/chemcrow_dark_bold.png">
  <img alt="ChemCrow logo" src="assets/chemcrow_light_bold.png" width="100%">
</picture>

## 1. Install

Run the commands below from the root of your downloaded or cloned repository.
Use Python 3.10; the package declares support for Python 3.9–3.11.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install this checkout to use its research features. Dependencies are defined in
[setup.py](setup.py); Git is needed for the pinned `paper-scraper` dependency.
The project uses the OpenAI 0.27.8, PaperQA 1.1.1, LangChain 0.0.275, and rmrkl
0.0.3 stack. Keep these versions together when setting up the environment.

## 2. Configure your services and papers

Create `.env` from [.env.example](.env.example) if you do not already have one:

```bash
# Preserve any existing configuration.
if [ ! -f .env ]; then cp .env.example .env; fi
mkdir -p local_papers
```

Edit `.env` and fill in `OPENAI_API_KEY`. For a first run with local PDFs, this is
the only required service key. Put your papers in `local_papers/`, or pass their
paths directly with `--pdf`.

| Setting | When to configure it |
| --- | --- |
| `OPENAI_API_KEY` | All CLI runs: agent, helper LLM, and PDF embeddings |
| `SEMANTIC_SCHOLAR_API_KEY` | Online paper discovery with `--source-mode combined` |
| `SERP_API_KEY` | Optional web discovery and the full-mode `WebSearch` tool |
| `CHEMSPACE_API_KEY` | Optional molecule pricing and molecular-identifier lookup fallback |
| `RXN4CHEM_API_KEY` + `RXN4CHEM_PROJECT_ID` | Optional remote reaction prediction and retrosynthesis; both are required |
| `RESEARCH_SOURCE_MODE` | Default literature sources: `local` or `combined` |
| `RESEARCH_PDF_DIR` | Default directory containing local PDFs; subdirectories are not searched |

The CLI loads `.env` beside `run_agent.py`; existing shell environment variables
have priority. Leave unused service keys blank. `.env`, local PDFs, model weights,
and generated results are ignored by Git; `.env.example` contains no credentials.

PDFs must contain extractable text and use a lowercase `.pdf` extension. Local
source mode skips online paper discovery, but document text still goes to the
configured hosted embeddings and LLMs.

**Local reaction models are optional.** You can follow all routes below without
installing them. Omit `--local-rxn`; remote reaction tools are available only if
both RXN settings are configured. Other chemistry tools remain available in full
mode. PubChem, Wikipedia, and patent-catalog access may use the network even
without a service key.

## 3. Choose a usage route

Execution mode and literature source are independent:

| Choice | What it controls |
| --- | --- |
| `--research` | Exposes only `ScientificResearch` and `SolutionDesign`; chemistry evaluation is deferred |
| `--source-mode local` | Retrieves literature from your PDFs |
| `--source-mode combined` | Adds online paper discovery; can also use your PDFs |

Source selection follows `--source-mode`, then `RESEARCH_SOURCE_MODE`, then
`local`. Local mode requires `--pdf`, `--pdf-dir`, or `RESEARCH_PDF_DIR`.

### Route A: Research and design from local papers

Start here to review papers and propose a solution without executing chemistry
tools. Replace `study.pdf` with your paper and the quoted task with your question.

```bash
python run_agent.py \
  "Use ScientificResearch to identify the stability limitations reported in this study. Use SolutionDesign to propose an evidence-grounded improvement and a plan for later evaluation. Cite the evidence and explain unresolved questions." \
  --research \
  --source-mode local \
  --pdf ./local_papers/study.pdf \
  --max-passages 6 \
  --max-evaluation-steps 12 \
  --max-iterations 20
```

Use `--pdf-dir ./local_papers` instead of `--pdf` to read a directory, or repeat
`--pdf` to select several files.

The intended sequence is:

```text
Your question + PDFs
    → ScientificResearch: retrieve passages and extract source-linked evidence
    → SolutionDesign: propose a solution and pending evaluation plan
    → Final answer: explain the proposal, sources, and unresolved checks
```

The plan may name chemistry tools for future use. Those tools cannot be executed
by the research-mode agent, and the plan does not establish that evaluation ran.

### Route B: Research using online literature

Configure `SEMANTIC_SCHOLAR_API_KEY`, then select `combined`. Configure
`SERP_API_KEY` as well if you want supplementary web discovery.

```bash
python run_agent.py \
  "Use ScientificResearch to investigate how surface passivation affects perovskite stability. Use SolutionDesign to propose a supported improvement and an evaluation plan. Cite sources and identify gaps." \
  --research \
  --source-mode combined \
  --max-passages 6 \
  --max-iterations 20
```

The tool searches for candidate papers, screens relevance, attempts PDF downloads,
retrieves passages, and extracts evidence. Add `--pdf-dir ./local_papers` to include
local papers in the same research run. Unavailable papers and retrieval failures
are reported as limitations; local papers remain usable if online discovery fails.

### Route C: Solve a task with the full toolset

Omit `--research` to let the agent execute chemistry tools. For a literature-based
task, explicitly request research, design, and evaluation:

```bash
python run_agent.py \
  "Use ScientificResearch to review the local papers for ways to improve molecular stability. Use SolutionDesign to propose a supported solution. Execute applicable checks with the available chemistry tools, compare actual observations with the proposal, and report unresolved checks." \
  --source-mode local \
  --pdf-dir ./local_papers \
  --max-passages 6 \
  --max-evaluation-steps 12 \
  --max-iterations 30
```

```text
Question → literature evidence → solution and plan
         → agent-selected chemistry tools → observations and final assessment
```

For a simple chemistry calculation, no literature workflow is necessary:

```bash
python run_agent.py \
  "Calculate the molecular weight of ethanol, SMILES CCO." \
  --source-mode combined \
  --max-iterations 10
```

Here `combined` avoids the CLI's local-PDF requirement; it does not force a paper
search. Tools are selected by the model. In both modes, the model can finish before
all requested stages run. Full mode does not automatically execute every saved
plan step. The separate [evaluation controller](chemcrow/evaluation.py) is a Python
API, not connected automatically to this CLI; see [its usage](docs/research.md).

### Common options

| Option | Use |
| --- | --- |
| `--model` | Agent and helper model; default `gpt-4.1-mini`. Use a model available to your account and compatible with the installed client |
| `--max-iterations` | Outer agent iteration limit; default 10. Research tools may make multiple internal calls per iteration |
| `--max-passages` | Retrieved passages per research call; 1–20, default 6 |
| `--max-evaluation-steps` | Maximum checks in the generated plan; 1–30, default 12. Does not execute the plan |
| `--trace` | A new JSON file for the agent trace; the file must not already exist |
| `--local-rxn` | Registers local reaction services on ports 8051/8052; use only after setting up those services |

Run `python run_agent.py --help` for the full argument list. Local reaction service
setup is documented separately in [chemcrow/docker/README.md](chemcrow/docker/README.md).
In research mode, selecting local reactions adds them only to the deferred plan's
capability catalog.

## 4. Read the results

The terminal prints the final answer, trace location, and counts of research and
solution-design calls. Files are created for the stages that actually run:

| File | What to read it for |
| --- | --- |
| `runs/<timestamp>.json` | Final answer, selected mode, available tools, actual tool inputs and observations, and call counts |
| `research_outputs/<run>/evidence.json` | Findings and source-reported limitations with titles, URLs, and page references |
| `research_outputs/<run>/literature_search.json` | Original retrieved passages supporting the evidence |
| `research_outputs/<run>/literature_selection.json` | Online paper selection and download outcomes, when applicable |
| `research_outputs/solution_<run>/solution_design.json` | Proposed solution, evidence references, pending evaluation steps, and unresolved checks |
| `research_outputs/solution_<run>/available_tools.json` | Tool capabilities used to generate that plan |

Start with the final answer and agent trace, then follow the returned evidence and
solution paths to inspect sources and planned checks. A saved plan remains a
planning artifact; actual agent tool observations are in `runs/`. Computational
predictions are not experimental measurements. Detailed JSON contracts are in
[docs/research.md](docs/research.md).

## 5. Where to modify the agent and tools

### Agent behavior and tool registration

| What you want to change | Edit here |
| --- | --- |
| CLI options, model default, PDF inputs, per-run limits, trace fields | [run_agent.py](run_agent.py), `main()` |
| Available tools, optional-service conditions, research/full mode catalogs | [chemcrow/agents/tools.py](chemcrow/agents/tools.py), `make_tools()` |
| Agent construction, model selection, executor configuration | [chemcrow/agents/chemcrow.py](chemcrow/agents/chemcrow.py), `ChemCrow` and `_make_llm()` |
| Full-mode reasoning and safety instructions | [chemcrow/agents/prompts.py](chemcrow/agents/prompts.py), `QUESTION_PROMPT` |
| Research-and-design-only instructions | [chemcrow/agents/prompts.py](chemcrow/agents/prompts.py), `RESEARCH_QUESTION_PROMPT` |
| Parsing model actions and final answers | [chemcrow/agents/output_parser.py](chemcrow/agents/output_parser.py) |
| Runtime dependencies or service-setting template | [setup.py](setup.py) and [.env.example](.env.example) |

### Tool implementations

The first column uses the names the agent sees. Some differ from their Python
class names. Edit a tool's `description` to change its usage guidance, and `_run()`
and its helper functions to change execution.

| Tool name | Implementation and modification point |
| --- | --- |
| `ScientificResearch` | [tools/research.py](chemcrow/tools/research.py): `load_local_papers()`, `retrieve_evidence()`, `build_evidence()`, and `scientific_research()` control PDF loading, passage retrieval, evidence extraction, and saved outputs; `ScientificResearch` holds tool settings and description |
| `SolutionDesign` | [tools/solution.py](chemcrow/tools/solution.py): `design_solution()` controls generation; schema classes and `_validate_design()` define valid plans; `tool_catalog()` defines the planner's capability list |
| `Name2SMILES`, `Mol2CAS`, `SMILES2Name` | [tools/converters.py](chemcrow/tools/converters.py): `Query2SMILES`, `Query2CAS`, and `SMILES2Name`; shared molecular/PubChem helpers are in [utils.py](chemcrow/utils.py) |
| `SMILES2Weight`, `MolSimilarity`, `FunctionalGroups` | [tools/rdkit.py](chemcrow/tools/rdkit.py): `SMILES2Weight`, `MolSimilarity`, and `FuncGroups` |
| `ControlChemCheck`, `SimilarityToControlChem`, `ExplosiveCheck`, `SafetySummary` | [tools/safety.py](chemcrow/tools/safety.py): tool classes and `MoleculeSafety`; summary prompts are in [tools/prompts.py](chemcrow/tools/prompts.py), and the runtime controlled-chemical table is [data/chem_wep_smi.csv](chemcrow/data/chem_wep_smi.csv) |
| `PatentCheck`, `WebSearch` | [tools/search.py](chemcrow/tools/search.py): corresponding classes and search helpers |
| `GetMoleculePrice` | [tools/chemspace.py](chemcrow/tools/chemspace.py): `GetMoleculePrice` and the `ChemSpace` client |
| `ReactionPredict`, `ReactionRetrosynthesis` — remote | [tools/rxn4chem.py](chemcrow/tools/rxn4chem.py): `RXNPredict`, `RXNRetrosynthesis`, and shared RXN client configuration |
| `ReactionPredict`, `ReactionRetrosynthesis` — local, optional | [tools/reactions.py](chemcrow/tools/reactions.py): `RXNPredictLocal` and `RXNRetrosynthesisLocal`; service implementations live under [docker/](chemcrow/docker/README.md) |
| `Python_REPL`, `Wikipedia` | Loaded from LangChain in [agents/tools.py](chemcrow/agents/tools.py); change registration there rather than editing an installed dependency |

### Literature backend and evaluation helpers

| Area | Edit here |
| --- | --- |
| Search-query generation, relevance-screening prompts, paper selection | [research_retrieval.py](chemcrow/research_retrieval.py) |
| Semantic Scholar requests, rate limits, retries, metadata cache | [semantic_scholar.py](chemcrow/semantic_scholar.py) |
| PDF downloads, timeouts, response validation | [paper_download.py](chemcrow/paper_download.py) |
| paper-scraper compatibility and PMC URL handling | [paperscraper_compat.py](chemcrow/paperscraper_compat.py) |
| Standalone plan execution, input checks, retries, assessments | [evaluation.py](chemcrow/evaluation.py), `evaluate_plan()` |

`LiteratureSearch` in `tools/search.py` and `GenerateResearchQuestions` in
[tools/hypothesis.py](chemcrow/tools/hypothesis.py) are legacy APIs, not registered
in the current agent. `hypothesis.py` also supplies evidence loading used by
solution design and evaluation.

When adding a tool, implement its name, description, and `_run()` in
`chemcrow/tools/`, then register it in `make_tools()`. Chemistry capabilities shared
with deferred research plans belong in `evaluation_tools`; tools exposed only in
full mode belong in that branch. Keep the research executor restricted to
`ScientificResearch` and `SolutionDesign`. If you rename a tool, update its
registration and any consumers of saved plans that use the old name.

For development, install `dev-requirements.txt` and run `python -m pytest -q`.
The repository retains the five original ChemCrow test files; external integration
tests are excluded by default.

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
