import re

import requests
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from chemcrow.input_utils import normalize_molecule_input

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def is_smiles(text):
    # This is a format probe: names and lookup errors are expected non-SMILES.
    # Do not let RDKit interpret trailing prose as a molecule name or log it.
    if not isinstance(text, str) or not text.strip() or re.search(r"\s", text.strip()):
        return False
    try:
        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(text.strip()) is not None
    except (ValueError, TypeError, RuntimeError):
        return False


def is_multiple_smiles(text):
    if is_smiles(text):
        return "." in text
    return False


def split_smiles(text):
    return text.split(".")


def is_cas(text):
    pattern = r"^\d{2,7}-\d{2}-\d$"
    return re.match(pattern, text) is not None


def largest_mol(smiles):
    ss = smiles.split(".")
    ss.sort(key=lambda a: len(a))
    while not is_smiles(ss[-1]):
        rm = ss[-1]
        ss.remove(rm)
    return ss[-1]


def canonical_smiles(smiles):
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        return smi
    except Exception:
        return "Invalid SMILES string"


def tanimoto(s1, s2):
    """Calculate the Tanimoto similarity of two SMILES strings."""
    try:
        mol1 = Chem.MolFromSmiles(s1)
        mol2 = Chem.MolFromSmiles(s2)
        fp1 = _MORGAN_GENERATOR.GetFingerprint(mol1)
        fp2 = _MORGAN_GENERATOR.GetFingerprint(mol2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except (TypeError, ValueError, AttributeError):
        return "Error: Not a valid SMILES string"


def pubchem_query2smiles(
    query: str,
    url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}",
) -> str:
    query = normalize_molecule_input(query)
    if is_smiles(query):
        if not is_multiple_smiles(query):
            return query
        else:
            raise ValueError(
                "Multiple SMILES strings detected, input one molecule at a time."
            )
    if url is None:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
    r = requests.get(url.format(query, "property/IsomericSMILES/JSON"))
    # convert the response to a json object
    data = r.json()
    # return the SMILES string
    try:
        properties = data["PropertyTable"]["Properties"][0]
        smi = properties.get("SMILES") or properties.get("IsomericSMILES")
    except (KeyError, IndexError, TypeError):
        raise ValueError(
            "Could not resolve a molecular structure. Provide one specific compound name "
            "or CAS number from the source; a chemical class such as heteropoly acid "
            "does not identify a unique molecule."
        ) from None
    if not is_smiles(smi):
        raise ValueError("Could not resolve a valid SMILES structure from the lookup response.")
    return str(Chem.CanonSmiles(largest_mol(smi)))


def query2cas(query: str, url_cid: str, url_data: str):
    query = normalize_molecule_input(query)
    try:
        mode = "name"
        if is_smiles(query):
            if is_multiple_smiles(query):
                raise ValueError(
                    "Multiple SMILES strings detected, input one molecule at a time."
                )
            mode = "smiles"
        url_cid = url_cid.format(mode, query)
        cid = requests.get(url_cid).json()["IdentifierList"]["CID"][0]
        url_data = url_data.format(cid)
        data = requests.get(url_data).json()
    except (requests.exceptions.RequestException, KeyError):
        raise ValueError("Invalid molecule input, no Pubchem entry")

    try:
        for section in data["Record"]["Section"]:
            if section.get("TOCHeading") == "Names and Identifiers":
                for subsection in section["Section"]:
                    if subsection.get("TOCHeading") == "Other Identifiers":
                        for subsubsection in subsection["Section"]:
                            if subsubsection.get("TOCHeading") == "CAS":
                                return subsubsection["Information"][0]["Value"][
                                    "StringWithMarkup"
                                ][0]["String"]
    except KeyError:
        raise ValueError("Invalid molecule input, no Pubchem entry")

    raise ValueError("CAS number not found")


def smiles2name(smi, single_name=True):
    """This function queries the given molecule smiles and returns a name record or iupac"""

    smi = normalize_molecule_input(smi)
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
    except Exception:
        raise ValueError("Invalid SMILES string")
    # query the PubChem database
    r = requests.get(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        + smi
        + "/synonyms/JSON"
    )
    # convert the response to a json object
    data = r.json()
    # return the SMILES string
    try:
        if single_name:
            index = 0
            names = data["InformationList"]["Information"][0]["Synonym"]
            while is_cas(name := names[index]):
                index += 1
                if index == len(names):
                    raise ValueError("No name found")
        else:
            name = data["InformationList"]["Information"][0]["Synonym"]
    except KeyError:
        raise ValueError("Unknown Molecule")
    return name
