"""Self-hosted reaction tools. Retrosynthesis, reaction forward prediction."""

import json

import requests
from langchain.tools import BaseTool

from chemcrow.utils import is_smiles

__all__ = ["RXNPredictLocal", "RXNRetrosynthesisLocal"]


class RXNPredictLocal(BaseTool):
    """Predict reaction."""

    name = "ReactionPredict"
    description = (
        "Predict the outcome of a chemical reaction. "
        "Takes as input the SMILES of the reactants separated by a dot '.', "
        "returns SMILES of the products."
    )

    def _run(self, reactants: str) -> str:
        """Run reaction prediction."""
        if not is_smiles(reactants):
            return "Incorrect input."

        product = self.predict_reaction(reactants)
        return product

    def predict_reaction(self, reactants: str) -> str:
        """Make api request."""
        try:
            response = requests.post(
                "http://localhost:8051/api/v1/run",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"smiles": reactants})
            )
            return response.json()['product'][0]
        except:
            return "Error in prediction."


class RXNRetrosynthesisLocal(BaseTool):
    """Predict retrosynthesis."""

    name = "ReactionRetrosynthesis"
    description = (
        "Predict a retrosynthetic route to a chemical compound. "
        "Takes product SMILES and returns proposed reaction steps as JSON. "
        "Does not provide experimental conditions or a validated procedure."
    )
    openai_api_key: str = ""
    endpoint: str = "http://localhost:8052/api/v1/retrosynth"

    def _run(self, reactants: str) -> str:
        """Run local retrosynthesis and return the first proposed route."""
        # Check that input is smiles
        if not is_smiles(reactants):
            return "Incorrect input."

        try:
            paths = self.retrosynthesis(reactants)
            if not paths:
                return "No retrosynthesis routes found for this target."
            return self.get_action_sequence(paths[0])
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            return f"Retrosynthesis error: {exc}"

    def retrosynthesis(self, reactants: str) -> list:
        """Make api request."""
        response = requests.post(
            self.endpoint,
            json={"target": reactants},
            timeout=(10, 240),
        )
        response.raise_for_status()
        paths = response.json()
        if not isinstance(paths, list) or any(not isinstance(p, dict) for p in paths):
            raise ValueError("Expected a list of route trees from the local service.")
        return paths

    def get_action_sequence(self, path):
        """Return route evidence for the main agent without another LLM call."""
        steps = self._preproc_actions(path)
        if not steps and not path.get("in_stock", False):
            raise ValueError("Route contains no mapped reactions and target is not in stock.")
        metadata = path.get("metadata", {})
        route_metadata = path.get("route_metadata", {})
        return json.dumps({
            "target": path["smiles"],
            "is_solved": metadata.get("is_solved", route_metadata.get("is_solved")),
            "target_in_stock": path.get("in_stock", False),
            "steps": steps,
            "note": (
                "Predicted route; steps are listed in forward synthesis order. "
                "Stock status refers to the configured catalog. "
                "Experimental conditions, quantities, yields, and validation are not provided."
            ),
        })

    def _preproc_actions(self, path):
        """Preprocess actions."""
        def _clean_actions(d):
            # A retrosynthesis tree starts at the final target. Visit precursor
            # branches first to report the steps in forward synthesis order.
            for child in d.get('children', []):
                yield from _clean_actions(child)
            if 'metadata' in d:
                if 'mapped_reaction_smiles' in d['metadata']:
                    product, reactants = d['metadata']['mapped_reaction_smiles'].split(">>")
                    if not product or not reactants:
                        raise ValueError("Route contains an empty reaction side.")
                    yield {"reactants": reactants, "products": product}

        rxns = list(_clean_actions(path))
        return rxns
