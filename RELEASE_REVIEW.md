# Cleanup and release review

The current checkout preserves the research pipeline and chemistry tools, fixes
CLI and tool-catalog regressions, and retains only the original five ChemCrow test
files at the owner's request. No commit, push, remote change, live API call, or
Docker startup was performed. Private inputs and outputs remain on disk.

## Repository structure

```text
chemcrow-public/
├── README.md, LICENSE, RELEASE_REVIEW.md
├── setup.py, MANIFEST.in, dev-requirements.txt, pytest.ini, .env.example
├── .github/workflows/         # tests.yml; archived build.yml.disabled
├── run_agent.py               # full agent or research-and-design-only CLI
├── chemcrow/
│   ├── agents/               # executor, registry, prompts, output parser
│   ├── tools/                # chemistry, research, solution and legacy APIs
│   ├── evaluation.py         # independent bounded evaluation controller
│   ├── research_retrieval.py  # paper queries, screening, retrieval orchestration
│   ├── semantic_scholar.py    # paced metadata client and cache
│   ├── paper_download.py      # bounded HTTP routes and PDF validation
│   ├── paperscraper_compat.py # PMC URL compatibility
│   ├── input_utils.py, utils.py
│   ├── data/                 # preserved chemical safety tables
│   ├── frontend/             # exported Streamlit callback compatibility
│   └── docker/               # optional reaction services and notices
├── tests/                    # test_agent.py, test_converters.py, test_rdkit.py,
│                             # test_safety_tools.py, test_search.py
├── docs/research.md           # workflow and artifact contracts
└── assets/                   # original logos and demo artwork
```

`.env`, local PDFs, `local_papers/`, `research_outputs/`, `runs/`, model weights,
installed metadata and caches are ignored. Do not force-add them.

## Behavior and fixes

- Removed the stale `args.local_retrieval` access that crashed the CLI after the
  owner replaced that flag with `--research`. Retrieval alone remains available
  through the `local_literature_search()` Python API.
- Source selection now follows explicit CLI arguments, environment settings,
  then the owner's `local` default. Local mode requires PDF inputs; combined mode
  supports online discovery. Invalid environment values fail before model setup.
- `--research` exposes only `ScientificResearch` and `SolutionDesign`. The planner
  retains a chemistry capability catalog for later checks. The executor cannot
  call those chemistry tools in this mode. Plans remain pending; research-mode
  prompts require identifying unresolved checks and unassessed safety.
- Full mode exposes the normal chemistry tools and configured optional services.
  Its planner catalog includes Python, Wikipedia and optional price/web-search
  tools, and matches the actual available evaluation tools.
- Restored explicit evidence-path/original-question handoff guidance to the
  owner's simplified prompt. Existing chemical safety instructions remain.
- Removed the hard-coded remote RXN project. Both `RXN4CHEM_API_KEY` and
  `RXN4CHEM_PROJECT_ID` are required for remote registration. Direct constructors
  accept `project_id` or the environment variable and reject a missing project.
  Local reaction services retain priority when selected.
- The remote route summary no longer adds an unsupported fixed solvent; it must
  use returned route information and report missing conditions as unknown.
- Archived the upstream automatic PyPI workflow as `build.yml.disabled`.
  GitHub release creation no longer activates that workflow.
- Updated README examples, configuration, testing commands, and research docs.

`ScientificResearch` saves original passages and source-linked evidence;
`SolutionDesign` saves an evidence-grounded proposal and pending evaluation plan.
The agent still chooses its calls and can finish early. The separate
`evaluation.evaluate_plan()` controller is preserved but is not automatically
connected to the agent or CLI. A plan is not evidence of executed checks.

Earlier conservative cleanup is retained: explicit registry imports, independent
API-key defaults, removal of an unused ClinTox constructor download, a shared
packaged safety-table loader, dependency declarations/pins, private-file ignores,
and source-distribution configuration. Safety data, checks, thresholds and output
messages remain. Original license, authorship and third-party notices remain.

## Test cleanup

The five original tracked test files remain, with external-service markers and
without automatic `.env` loading. The additional 12 test modules and `conftest.py`
were backed up outside the repository and removed. Test bytecode was removed.
No replacement tests or fixtures were added elsewhere in the repository.

`pytest.ini` excludes `integration` tests by default. There is no permanent socket
blocker or `--run-integration` option after removing `conftest.py`. CI now covers
only the original tests; research-specific regression coverage is no longer part
of the committed suite. The temporary checks below were run outside the repository.

```bash
python -m pytest -q
# Only when external services and credentials are intentionally configured:
python -m pytest -o addopts='' -m integration
```

## Executed validation

Validation used the existing Python 3.10.22 ChemCrow environment.

| Check | Result |
| --- | --- |
| Corrected suite before removing additional tests | **193 passed, 17 skipped, 1 deselected**; the deselected test targeted the deliberately removed retrieval-only CLI flag |
| Temporary checks outside the repository | **12 passed**: actual CLI/agent handoff in both modes, local/remote capability catalogs, source precedence, PDF/PaperQA/FAISS retrieval with fake hosted models, RXN configuration and summary constraints |
| Final five-file suite | **14 passed, 3 skipped, 14 deselected**, one dependency deprecation warning |
| Final-suite isolation | Temporary external pytest plugin blocked sockets and disabled credential loading; no plugin retained in the repository |
| CLI help | Passed; current research and source options are listed |
| Python syntax and installed dependencies | All 38 release Python files parsed; `pip check` found no broken requirements |
| Temporary source and wheel builds | Passed; 87 source-archive members and 35 wheel members; only the original five tests in the source archive |
| Archive contents | Current core modules and safety data included; no private inputs, outputs, credentials or caches |
| Private-file preservation | All 172 files in the pre-fix input/output/credential snapshot unchanged, untracked and ignored |
| Local documentation links and whitespace | Passed |

The three skipped cases are the original explicitly disabled hosted-LLM tests.
The 14 deselected cases require external services or downloaded catalogs.
Temporary checks used fake LLMs, embeddings and service clients; they establish
code behavior, not the correctness or availability of live services.

Live OpenAI, Semantic Scholar, publisher downloads, SerpAPI, PubChem, ChemSpace,
RXN4Chem, Wikipedia, catalog downloads and Docker services were not tested.
No fresh network dependency resolution or Python 3.9/3.11 matrix run was performed.

## Publication considerations

The previous cleanup inspected the working tree and locally available Git history
(142 commits, 327 blobs), including common token patterns and exact matches to
local credential values. No credential matches were found; values were not
printed. This is a bounded local scan, not a guarantee about unknown secret
formats or remote-only history. Research outputs may contain private text and
absolute paths and remain ignored.

After these fixes, all 69 current release candidate files were checked again for
local credential values and common token/private-key patterns; no matches were found.

The fixed RXN project/solvent behavior and automatic PyPI workflow identified in
the earlier review are now addressed. A GitHub source upload can be prepared after
reviewing the intended diff and choosing the owner's repository. Many research
files are untracked and must be explicitly included in that commit. The current
remote and package metadata still identify upstream; no remote was changed.

Fresh installation across supported Python versions and live integration checks
remain outstanding. Dependencies are not fully locked. A PyPI release needs its
own package-name and VCS-dependency decisions. Review third-party data/model terms
before redistribution. Legacy helpers still contain broad exception handling and
some unbounded HTTP calls; provenance validation does not prove scientific
entailment, and model predictions do not establish experimental success or safety.
