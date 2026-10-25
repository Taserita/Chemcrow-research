import os

from langchain import agents
from langchain.base_language import BaseLanguageModel

from chemcrow.tools.chemspace import GetMoleculePrice
from chemcrow.tools.converters import Query2CAS, Query2SMILES, SMILES2Name
from chemcrow.tools.rdkit import FuncGroups, MolSimilarity, SMILES2Weight
from chemcrow.tools.reactions import RXNPredictLocal, RXNRetrosynthesisLocal
from chemcrow.tools.research import ScientificResearch
from chemcrow.tools.rxn4chem import RXNPredict, RXNRetrosynthesis
from chemcrow.tools.safety import (
    ControlChemCheck,
    ExplosiveCheck,
    SafetySummary,
    SimilarControlChemCheck,
)
from chemcrow.tools.search import PatentCheck, WebSearch
from chemcrow.tools.solution import SolutionDesign, tool_catalog


def make_tools(
    llm: BaseLanguageModel,
    api_keys: dict = None,
    local_rxn: bool = False,
    verbose=True,
    research: bool = False,
    source_mode: str = None,
):
    api_keys = api_keys or {}
    source_mode = source_mode or os.getenv("RESEARCH_SOURCE_MODE") or "local"

    serp_api_key = api_keys.get("SERP_API_KEY") or os.getenv("SERP_API_KEY")
    rxn4chem_api_key = api_keys.get("RXN4CHEM_API_KEY") or os.getenv("RXN4CHEM_API_KEY")
    rxn4chem_project_id = api_keys.get("RXN4CHEM_PROJECT_ID") or os.getenv("RXN4CHEM_PROJECT_ID")
    openai_api_key = api_keys.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    chemspace_api_key = api_keys.get("CHEMSPACE_API_KEY") or os.getenv("CHEMSPACE_API_KEY")
    semantic_scholar_api_key = (
        api_keys.get("SEMANTIC_SCHOLAR_API_KEY")
        or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    )

    # Research and solution-design tools
    research_tool = ScientificResearch(
        llm=llm,
        serp_api_key=serp_api_key,
        openai_api_key=openai_api_key,
        semantic_scholar_api_key=semantic_scholar_api_key,
        local_pdf_dir=os.getenv("RESEARCH_PDF_DIR"),
        source_mode=source_mode,
    )

    solution_tool = SolutionDesign(llm=llm)

    evaluation_tools = [
        Query2SMILES(chemspace_api_key),
        Query2CAS(),
        SMILES2Name(),
        PatentCheck(),
        MolSimilarity(),
        SMILES2Weight(),
        FuncGroups(),
        ExplosiveCheck(),
        ControlChemCheck(),
        SimilarControlChemCheck(),
        SafetySummary(llm=llm),
    ]

    if local_rxn:
        evaluation_tools.extend([
            RXNPredictLocal(),
            RXNRetrosynthesisLocal(),
        ])
    elif rxn4chem_api_key and rxn4chem_project_id:
        evaluation_tools.extend([
            RXNPredict(rxn4chem_api_key, project_id=rxn4chem_project_id),
            RXNRetrosynthesis(
                rxn4chem_api_key, openai_api_key, project_id=rxn4chem_project_id,
            ),
        ])

    if research:
        # Chemistry capabilities describe deferred checks; they are not executable
        # by this restricted agent.
        all_tools = [research_tool, solution_tool]
    else:
        # Preserve the original ChemCrow tool registration.
        all_tools = agents.load_tools(["python_repl", "wikipedia"])

        all_tools.extend([
            research_tool,
            solution_tool,
            *evaluation_tools,
        ])

        if chemspace_api_key:
            all_tools.append(GetMoleculePrice(chemspace_api_key))

        if serp_api_key:
            all_tools.append(WebSearch(serp_api_key))


    solution_tool.available_tools = tool_catalog(evaluation_tools if research else all_tools)

    return all_tools
