"""Compatibility fixes for the paper-scraper version used by ChemCrow."""

from urllib.parse import urljoin

import paperscraper
from aiohttp import ClientResponseError
from paperscraper import lib
from paperscraper.exceptions import NoPDFLinkError
from paperscraper.utils import search_pdf_link


async def find_pmc_pdf_link(pmc_id, session):
    """Resolve PDF links against the article URL, including HTTP redirects."""
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
    async with session.get(url) as response:
        try:
            response.raise_for_status()
        except ClientResponseError as exc:
            raise RuntimeError(
                f"Failed to download PubMed Central ID {pmc_id} from URL {url}."
            ) from exc
        try:
            pdf_link = search_pdf_link(text=await response.text())
        except NoPDFLinkError as exc:
            raise RuntimeError(
                f"No PDF link matched for PubMed Central ID {pmc_id} from URL {url}."
            ) from exc
        return urljoin(str(response.url), pdf_link)


def search_papers(*args, **kwargs):
    """Search with the PMC resolver fix shared by PMC and PubMed downloads.

    The installed dependency concatenates a hostname with relative PDF links,
    producing invalid hosts such as ``www.ncbi.nlm.nih.govpdf``. Replace only
    that helper at runtime; leave the installed package files untouched.
    """
    lib.find_pmc_pdf_link = find_pmc_pdf_link
    return paperscraper.search_papers(*args, **kwargs)
