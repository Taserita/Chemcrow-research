from setuptools import find_packages, setup

exec(open("chemcrow/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chemcrow",
    python_requires=">=3.9, <3.12",  # this temporarily fixes molbloom install, which breaks with 3.12
    version=__version__,
    description="Accurate solution of reasoning-intensive chemical tasks, powered by LLMs.",
    author="Andres M Bran, Sam Cox, Andrew White, Philippe Schwaller",
    author_email="andrew.white@rochester.edu",
    url="https://github.com/ur-whitelab/chemcrow-public",
    license="MIT",
    packages=find_packages(),
    package_data={"chemcrow": ["data/chem_wep_smi.csv"]},
    install_requires=[
        "python-dotenv",
        "rdkit",
        "openai==0.27.8",
        "molbloom",
        "paper-qa==1.1.1",
        "google-search-results",
        "langchain==0.0.275",
        "nest_asyncio",
        "tiktoken",
        "rmrkl==0.0.3",
        "paper-scraper @ git+https://github.com/blackadad/paper-scraper.git@29c11f081caec38834b2e8b022e078c77f82975a",
        "aiohttp",
        "pandas",
        "pydantic>=1,<3",
        "pypdf",
        "PyMuPDF",
        "requests",
        "streamlit",
        "rxn4chemistry",
        "wikipedia",
    ],
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
