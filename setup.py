from setuptools import setup, find_packages

setup(
    name="aPTR",
    version="0.0.1",
    description="Microbial dynamics inferred from single-sample amplicon sequencing",
    author="Philippe Chlenski",
    author_email="pac@cs.columbia.edu",
    url="http://www.github.com/pchlenski/aptr",
    # packages=find_packages(),
    # packages=["aptr"],
    packages=find_packages(where="aptr", exclude=["tests"]),
    package_dir={"aptr": "aptr"},
    install_requires=[
        "biopython",
        "numpy",
        "pandas",
        "uuid",
        "torch",
        "matplotlib",
    ],
)
