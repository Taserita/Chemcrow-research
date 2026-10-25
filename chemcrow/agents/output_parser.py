"""ReAct parser that preserves inputs before removing wrapper quotes."""

import re

from langchain.schema import AgentAction, OutputParserException
from rmrkl.output_parser import ChatZeroShotOutputParser
from rmrkl.prompts import FINAL_ANSWER_ACTION

from chemcrow.input_utils import normalize_tool_input


class ChemCrowOutputParser(ChatZeroShotOutputParser):
    def parse(self, text):
        if FINAL_ANSWER_ACTION in text:
            return super().parse(text)

        # Parse the original input: rmrkl strips opening quotes before newlines,
        # which turns e.g. '"aspirin"\n' into 'aspirin"\n'.
        match = re.search(
            r"Action\s*\d*\s*:\s*(.*?)\s*Action\s*\d*\s*Input\s*\d*\s*:\s*(.*)",
            text,
            re.DOTALL,
        )
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")
        action = match.group(1).strip()
        action_input = normalize_tool_input(match.group(2))
        if not action or not action_input:
            raise OutputParserException("Action and Action Input must not be empty.")
        return AgentAction(action, action_input, text)
