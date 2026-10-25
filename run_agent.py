import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from chemcrow.agents import ChemCrow
from chemcrow.tools.research import ScientificResearch
from chemcrow.agents.tools import make_tools
from chemcrow.tools.solution import SolutionDesign


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one ChemCrow task.")
    parser.add_argument("task", help="Chemistry task or research and solution-design question")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model to use for the agent")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument(
        "--research", action="store_true",
        help="Restrict the agent to ScientificResearch and SolutionDesign; defer chemistry evaluation",
    )
    parser.add_argument("--source-mode", choices=["local", "combined"], default=None,
                        help="Sources for ScientificResearch (default: RESEARCH_SOURCE_MODE or local)")
    parser.add_argument("--pdf", action="append", type=Path, default=[],
                        help="Local PDF for research (repeat for multiple files)")
    parser.add_argument("--pdf-dir", type=Path,
                        help="Directory of local PDFs for retrieval (nonrecursive)")
    parser.add_argument("--max-passages", type=int, default=6,
                        help="Maximum retrieved passages, from 1 to 20 (default: 6)")
    parser.add_argument("--max-evaluation-steps", type=int, default=12,
                        help="Maximum checks in a SolutionDesign plan (1 to 30)")
    parser.add_argument(
        "--local-rxn", action="store_true",
        help="Use local reaction services on ports 8051 and 8052",
    )
    parser.add_argument(
        "--trace", type=Path,
        help="Save the agent trace to this new JSON file (default: runs/)",
    )
    args = parser.parse_args(argv)
    if not args.task.strip():
        parser.error("Provide a nonempty task.")

    if not 1 <= args.max_passages <= 20:
        parser.error("--max-passages must be between 1 and 20.")
    if not 1 <= args.max_evaluation_steps <= 30:
        parser.error("--max-evaluation-steps must be between 1 and 30.")

    # Load .env from the same directory as this script.
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args.source_mode = args.source_mode or os.getenv("RESEARCH_SOURCE_MODE") or "local"
    if args.source_mode not in ("local", "combined"):
        parser.error("RESEARCH_SOURCE_MODE must be local or combined.")
    if args.source_mode == "local" and not (args.pdf or args.pdf_dir):
        if os.getenv("RESEARCH_PDF_DIR"):
            args.pdf_dir = Path(os.environ["RESEARCH_PDF_DIR"])
        else:
            parser.error("Local sources require --pdf or --pdf-dir.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in your environment or .env")

    helper_llm = ChatOpenAI(
        model_name=args.model, openai_api_key=api_key,
        temperature=0, streaming=False,
    )
    tools = make_tools(
        helper_llm, api_keys={"OPENAI_API_KEY": api_key},
        local_rxn=args.local_rxn, research=args.research, source_mode=args.source_mode,
    )
    for tool in tools:
        if isinstance(tool, ScientificResearch):
            if args.source_mode:
                tool.source_mode = args.source_mode
            if args.pdf:
                tool.local_pdf_paths = [str(path.resolve()) for path in args.pdf]
            if args.pdf_dir:
                tool.local_pdf_dir = str(args.pdf_dir.resolve())
            tool.max_passages = args.max_passages
            tool.description += (
                f" Configured source mode: {tool.source_mode}."
                f" Local PDFs: {json.dumps(tool.local_pdf_paths)}."
                f" Local PDF directory: {tool.local_pdf_dir or 'none'}."
            )
        elif isinstance(tool, SolutionDesign):
            tool.user_question = args.task
            tool.max_evaluation_steps = args.max_evaluation_steps

    agent = ChemCrow(
        model=args.model,
        tools_model=args.model,
        openai_api_key=api_key,
        temp=0.1,
        max_iterations=args.max_iterations,
        streaming=False,
        local_rxn=args.local_rxn,
        tools=tools,
    )

    started_at = datetime.now(timezone.utc)
    trace_path = args.trace or Path("runs") / started_at.strftime("%Y%m%dT%H%M%S_%fZ.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nTask:", args.task)
    print("Model:", args.model)
    # ChemCrow.run() returns only the final answer. Ask the executor for
    # intermediate actions and observations as well.
    agent.agent_executor.return_intermediate_steps = True
    result = agent.agent_executor({"input": args.task})
    answer = result["output"]

    trace = {
        "mode": "scientific_research_agent",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "model": args.model,
        "tools_model": args.model,
        "max_iterations": args.max_iterations,
        "local_rxn": args.local_rxn,
        "max_evaluation_steps": args.max_evaluation_steps,
        "research_only": args.research,
        "source_mode": args.source_mode,
        "available_tools": [tool.name for tool in tools],
        "steps": [
            {
                "step": index,
                "tool": action.tool,
                "tool_input": action.tool_input,
                "observation": observation,
                "model_output": action.log,
            }
            for index, (action, observation) in enumerate(
                result["intermediate_steps"], start=1
            )
        ],
        "final_answer": answer,
    }
    trace["scientific_research_calls"] = sum(
        step["tool"] == "ScientificResearch" for step in trace["steps"]
    )
    trace["solution_design_calls"] = sum(
        step["tool"] == "SolutionDesign" for step in trace["steps"]
    )
    with trace_path.open("x", encoding="utf-8") as trace_file:
        json.dump(trace, trace_file, indent=2, ensure_ascii=False, default=str)
        trace_file.write("\n")

    print("\nFINAL ANSWER")
    print(answer)
    print("\nTrace saved to:", trace_path.resolve())
    print("ScientificResearch calls:", trace["scientific_research_calls"])
    print("SolutionDesign calls:", trace["solution_design_calls"])


if __name__ == "__main__":
    main()
