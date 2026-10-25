# Research workflow and artifact contracts

## Scientific research evidence

`ScientificResearch` in `chemcrow/tools/research.py` runs three stages:

1. Web discovery using SerpAPI, retaining titles, links, and snippets.
2. Literature retrieval using the existing paper-scraper and PaperQA stack.
3. LLM extraction of findings and source-reported limitations into local JSON.

The tool is registered automatically by `make_tools()`. For example, with the
local runner:

```bash
python run_agent.py \
  "Use ScientificResearch to investigate how surface passivation affects perovskite stability. Save findings and limitations for solution design." \
  --source-mode combined
```

It uses the configured helper LLM. `OPENAI_API_KEY` is needed for the default
hosted LLM and embeddings. `SERP_API_KEY` enables web discovery;
`SEMANTIC_SCHOLAR_API_KEY` is passed to the paper search backend. If the SerpAPI
key is missing or web discovery fails, paper retrieval is still attempted.

Each invocation creates a unique directory under `research_outputs/`:

```text
research_outputs/<timestamp>_<run-id>/
    web_search.json          # discovery results
    literature_search.json   # original passages, paper metadata, page/chunk locators
    evidence.json            # research_question, evidence, limitations, open_questions
    extraction.json          # extraction responses and validation errors, when extraction runs
    trace.json               # internal stage status and timestamps
    papers/                  # downloaded documents, when available
```

The tool returns the absolute `evidence_path` and the evidence packet. The
evidence file contains exactly `id`, `category`, `claim`, `source_title`,
`source_url`, and `source_locator` per new item. Categories are `finding` and
`limitation`; both summarize relevant source content. There is no separate
experimental-results extraction or generated research-question step during
evidence extraction. Source-reported limitations are evidence entries with page
references; the top-level `limitations` list records retrieval/extraction caveats,
and `open_questions` remains empty for compatibility with existing consumers.
The internal `source_id` links model output to retrieved passages but is not
included in saved evidence entries. The LLM summarizes retrieved original document
text in at most 600 characters and 100 words per claim. Titles, URLs, and PDF
page ranges (for example `P1 pages 4-6`) are copied by the program, not generated
by the model. These ranges identify the retrieved chunk, not an exact sentence
location. Original passages remain in `literature_search.json` for review.
New evidence does not require verbatim quotations or an `evidence_type` label.
Invalid entries receive one correction attempt without discarding valid summaries.
Source-ID checks do not verify that a summary is scientifically supported.
Solution design accepts both this format and older evidence files containing
`source_passage` and `evidence_type`.

The constructor accepts `output_dir`, `max_papers` (default 5),
`max_passages` (6), and `max_evidence` (8). The paper count is a retrieval target;
online research screens at most 30 unique candidates across up to 6 metadata
requests and attempts at most 10 paper downloads. This workflow has internal
calls beyond the outer agent's iteration count.

In combined mode, `ScientificResearch` requires `SEMANTIC_SCHOLAR_API_KEY` for
online paper discovery. Calls with the same key share a process-local client:
request starts are spaced at least 1.2 seconds apart, successful metadata responses
are cached for one hour (up to 128 entries), and HTTP 429 retries respect
`Retry-After`. Each request permits at most five attempts and 120 seconds of
scheduled waiting; server cooldowns persist even after a failed call. Separate
processes and the legacy `LiteratureSearch` do not share this limiter.

Before downloading, the LLM screens titles and abstracts against the research
question, using three labels: `direct`, `relevant`, and `irrelevant`. Both `direct`
and `relevant` papers qualify for download, with direct matches prioritized within
each batch. Related methods and background can qualify as relevant. Only the paper
ID and relevance label are required; reasons and metadata quotes are optional.
Missing abstracts do not automatically disqualify a paper. Screening is an LLM
judgment, not proof that a paper will contain usable evidence. Final evidence claims
still require matching retrieved source passages.

For each eligible paper, ScientificResearch first downloads `openAccessPdf.url`,
then tries paper-scraper's alternative sources if necessary. Responses are reused
across routes, including known redirect targets. An HTML landing page may be parsed
for its PDF link without fetching the same page again. Publisher sessions never
receive the Semantic Scholar API key. Requests to each publisher host are spaced
one second apart; transient failures allow one retry. Requests time out after 30
seconds, each download route has a 60-second timeout, and the network budget per
paper is 180 seconds. Individual response bodies are limited to 50 MiB.

Each route writes a temporary file. Before marking a paper `downloaded`, the tool
checks the PDF signature, verifies the file is a PDF, and reads its pages with
pypdf to confirm extractable text. HTML challenge pages and malformed PDFs are
rejected; a PDF with no text is recorded as `no_text` if no usable alternative is
found. Only validated files are moved into the final PDF path and passed to PaperQA.
The selection record includes `openAccessPdf`, `isOpenAccess`, and `download_attempts`
with source method, requested/redirected URLs, HTTP status, content type, cache use,
validation result, and errors. Qualified papers left unattempted receive a budget
or target reason. This does not guarantee publisher access or add OCR support.

Each combined-mode run saves `literature_selection.json` alongside `evidence.json`,
with attempted queries, candidate metadata, relevance decisions/reasons, download
outcomes, and retrieval limitations. Each `screening_XX.json` saves the input metadata,
raw model responses, detailed validation errors, and accepted decisions. Validation
is per paper: valid decisions survive failures elsewhere in the batch, and unresolved
IDs are retried once. Persistent failures have `screening_status: "failed"` and
`relevance: null`, rather than being mislabeled as irrelevant. The tool returns
`selection_path`. The evidence
schema, local-only retrieval, and legacy `LiteratureSearch` behavior are unchanged.

Failures and unavailable evidence are recorded in `limitations`; a run with
no usable evidence saves an empty evidence array and returns `no_evidence`.
Hypothesis generation is not implemented by this tool.
Invalid extraction JSON or schema triggers one regeneration attempt with
validation feedback. Both attempts are saved in `extraction.json` for diagnosis;
regenerated claims still undergo the same source-passage checks.

To combine local PDFs with downloaded papers, configure the research tool:

```python
from chemcrow.tools.research import ScientificResearch

# llm is your configured LangChain helper LLM.
tool = ScientificResearch(
    llm=llm,
    local_pdf_dir="./papers",             # optional, nonrecursive directory
    local_pdf_paths=["./local_papers/study.pdf"],    # optional, explicit files
)
result = tool.run("What limitations and unresolved questions do these studies report?")
```

For the existing agent runner, set `RESEARCH_PDF_DIR` before running it:

```bash
export RESEARCH_PDF_DIR=./local_papers
python run_agent.py "Use ScientificResearch to investigate surface passivation and stability."
```

Local and downloaded papers enter the same PaperQA `Docs` collection before
retrieval. Local papers remain usable when online paper search fails or finds
no papers. `max_papers` limits downloaded candidates; all configured local PDFs
are attempted. `max_passages` limits retrieval across the combined collection.
Local PDFs must have a readable text layer and a lowercase `.pdf` extension;
OCR is not included. Failed files are recorded in `limitations`.

Local titles come from PDF metadata, falling back to the filename. Local
`source_url` values are absolute `file://` URIs; page references are preserved
in `literature_search.json`. Identical local files, including copies already
downloaded, are added only once. The evidence JSON schema is unchanged.

The reusable `load_local_papers(docs, pdf_paths, sources=None, pdf_dir=None)`
function also adds PDFs directly to an existing PaperQA collection and returns
the source mapping and limitations. Pass the returned mapping into later calls
to retain source attribution. This change does not persist the vector index
between runs. Hosted embeddings and LLM calls still process document text.

To retrieve from **local papers only**, set `source_mode="local"` on
`ScientificResearch`. It skips both SerpAPI and online paper downloads, retrieves
relevant PDF passages, and saves the same evidence and retrieval JSON files:

```python
tool = ScientificResearch(
    llm=llm,
    source_mode="local",
    local_pdf_paths=["./local_papers/study.pdf"],
    max_passages=6,
)
result = tool.run("What limitations and open questions does this paper identify?")
```

For the agent runner, set `RESEARCH_SOURCE_MODE=local` together with
`RESEARCH_PDF_DIR`. This selects sources for `ScientificResearch`; other agent
tools retain their normal behavior. The CLI and `make_tools()` resolve an explicit
source mode first, then `RESEARCH_SOURCE_MODE`, and otherwise use `local`. The CLI
requires local PDF inputs in local mode. Constructing `ScientificResearch` directly
without `source_mode` still defaults to `combined`.

For retrieval without evidence extraction, call
`local_literature_search(question, llm, local_pdf_paths=[...])`. To query an
already loaded collection repeatedly, use the source mapping from the loader:

```python
from chemcrow.tools.research import load_local_papers, retrieve_evidence

loaded = load_local_papers(docs, ["./local_papers/study.pdf"])
retrieval = retrieve_evidence(
    "What evaluation methods are reported?", docs, loaded["sources"], max_passages=4,
)
print(retrieval["passages"])
```

PaperQA embeds the question, searches its vector index, and filters retrieved
chunks using the helper LLM. Returned passages contain original text, source
titles, file URIs, and page references. Reusing the same `docs` instance reuses
its in-memory index. Local mode still uses the configured hosted embeddings and
LLM; it is not a fully offline mode.

To run **literature-assisted problem solving**, use the full agent:

```bash
python run_agent.py \
  "Use ScientificResearch to review the local aspirin papers. Use SolutionDesign to propose an evidence-grounded route and a tool-based evaluation plan. Execute the available checks, compare their actual outputs with the plan, and report support and unresolved questions." \
  --source-mode local --pdf-dir ./local_papers/aspirin \
  --max-passages 9 --max-evaluation-steps 12 --max-iterations 20 --local-rxn
```

The full tool set includes `ScientificResearch`, `SolutionDesign`, molecular tools,
plus configured reaction tools. `--local-rxn` enables services at ports 8051/8052;
these services must be running for reaction checks. Omit it to use configured
remote reaction tools (both `RXN4CHEM_API_KEY` and `RXN4CHEM_PROJECT_ID` are required)
or proceed without reaction capabilities. In full mode, the planner sees the
actual registered evaluation tools, including Python, Wikipedia, and optional
price/web-search tools. Registration does not establish service health.
Use `--source-mode combined` for online discovery, and `--pdf`/`--pdf-dir` for local
papers. Add `--research` to restrict this run to literature research and design.

The LLM chooses when to retrieve evidence, design a solution, execute tools, and
conclude. These stages are prompted, not enforced by the agent executor.
`SolutionDesign` saves a pending plan; it does not automatically execute it.

Traces in `runs/` contain actual agent actions and observations, the final answer,
`research_only`, `source_mode`, and `scientific_research_calls` and
`solution_design_calls`. They do not contain
controller evaluation paths or step tags. `--trace` selects a new trace file.

To perform **only literature research and solution design**, use:

```bash
python run_agent.py \
  "Review this study and design an evidence-grounded solution and evaluation plan." \
  --research --pdf ./local_papers/study.pdf --max-passages 4 --max-iterations 20
```

Repeat `--pdf` for multiple papers or use `--pdf-dir ./papers` for a directory.
This mode still uses the agent loop, but exposes only `ScientificResearch` and
`SolutionDesign`. Its planner retains a chemistry-tool catalog for later evaluation;
ChemCrow does not replace that catalog with the restricted two-tool list. Planned
checks remain pending and cannot be executed by this research agent. `--local-rxn`
adds local reaction capabilities to that deferred catalog without calling services.
The prompt requires reporting unresolved checks rather than claiming validation.
The model may still finish before calling both research and design tools.

The previous retrieval-only CLI flag has been removed. For retrieval without
agent selection or evidence extraction, the Python API remains available:

```python
from chemcrow.tools.research import local_literature_search

# llm is your configured LangChain helper LLM.
result = local_literature_search(
    "What limitations does this study report?",
    llm=llm,
    local_pdf_paths=["./local_papers/study.pdf"],
    max_passages=4,
)
print(result["passages"])
```

Hosted embeddings and the helper LLM still process the PDF text. Agent traces
are written under `runs/`, or to the new file selected by `--trace`.

## Evidence-grounded solution design

`SolutionDesign` in `chemcrow/tools/solution.py` replaces `GenerateResearchQuestions`
in the active tool registry. The old `hypothesis.py` API remains available for
legacy callers; it is not exposed to the agent. Existing evidence files remain
readable. The tool designs one solution and an ordered evaluation plan rather
than generating research-question candidates.

The tool accepts an evidence file path, or a JSON string containing `evidence_file`
and `question`. The runner configures the original user task, so a bare path keeps
that task distinct from the narrower literature search question. Direct Python use:

```python
import json
from chemcrow.tools.solution import SolutionDesign, tool_catalog

# evaluation_tools are the same tool instances available to the executing agent.
planner = SolutionDesign(llm=llm, available_tools=tool_catalog(evaluation_tools))
result = json.loads(planner.run(json.dumps({
    "evidence_file": "/absolute/path/to/evidence.json",
    "question": "Can the proposed route produce aspirin?"
})))
```

Each run saves:

```text
research_outputs/solution_<timestamp>_<run-id>/
    solution_design.json    # solution, evaluation_plan, unresolved_checks, limitations
    evidence_snapshot.json  # validated input evidence
    available_tools.json    # tool names and descriptions used for planning
    generation.json         # raw model outputs and validation errors
    trace.json              # planning status, timestamps, evaluation step count
```

`solution` contains a summary, evidence-linked steps, expected outcomes, and
assumptions. Each `evaluation_plan` entry includes its ID, solution step references,
exact tool name, purpose, `tool_input`, `input_instructions`, earlier `depends_on`
IDs, and `success_criteria`. Use a null `tool_input` for inputs that must be resolved
from previous observations. Each saved check starts with `status: pending`; the
packet has `evaluation_status: not_started`. This is a planning artifact, not an
execution report. Agent tool calls are recorded in the agent trace. A separately invoked controller writes its own evaluation artifact; the original plan is preserved.

Unknown evidence/tool/step references, duplicate IDs, invalid dependencies, and
schema errors trigger one correction attempt. Repeated invalid output returns
`failed` with no executable plan. Empty evidence returns `no_solution` without
calling the LLM. A supported proposal returns `designed`, not a claim of successful
validation. When suitable tools are unavailable, unsupported checks must appear
in `unresolved_checks`; the plan may be empty. `--max-evaluation-steps` defaults
to 12 and permits 1–30 checks; it does not set the agent's iteration budget.

`SolutionDesign` returns the proposal without executing it. The independent
controller in `chemcrow/evaluation.py` is implemented and tested, but is not
connected to `ChemCrow` or `run_agent.py`. When called directly, the controller:

- Loads the saved plan and evidence snapshot, then selects dependency-ready checks.
- Uses concrete planned inputs directly; asks the LLM for one structured input
  when a value depends on earlier observations or needs repair.
- Validates the tool name and input, performs the real call, and saves its response
  before asking the LLM to assess it against the criteria.
- Retries input/call failures within the configured limit. Failed dependencies
  block dependent checks while independent checks continue.
- Finalizes only when every step is `completed`, `failed`, or `blocked`. A completed
  call can have a `supports`, `disagrees`, or `inconclusive` assessment.

Each execution creates a fresh subdirectory beside the plan:

```text
evaluation_<timestamp>_<id>/
    solution_snapshot.json
    evidence_snapshot.json
    evaluation.json        # incremental statuses, inputs, full observations, assessments
    final_answer.md        # conclusion plus program-generated accounting of every step
```

The controller defaults to two attempts per step, 120 seconds per call, 900
seconds total, and at most 60 actual tool calls. Budget exhaustion is recorded;
unattempted checks are blocked. Its synchronous deadlines require a POSIX main
thread. A process crash can leave a persisted step running; each new invocation
creates a fresh evaluation directory.

To call the existing Python API with configured tools and an LLM:

```python
from chemcrow.evaluation import evaluate_plan

result = evaluate_plan(
    "research_outputs/solution_<run>/solution_design.json",
    tools=tools, llm=llm,
    max_attempts=2, call_timeout=120, total_timeout=900,
)
print(result["final_answer"])
```

There are no `--evaluate-plan`, `--evaluation-attempts`, `--evaluation-timeout`,
`--evaluation-budget`, or `ChemCrow(auto_evaluate=...)` options in this checkout.
`--max-evaluation-steps` only bounds plan size.

The controller's final text includes program-generated accounting of every step.
Assessment and interpretation still depend on the LLM; execution accounting does
not prove scientific correctness. Reaction predictions are computational checks,
not physical experiments or measured yield/purity.
