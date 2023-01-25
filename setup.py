from setuptools import setup

setup(
    name="aPTR",
    version="0.0.1",
    description="Microbial dynamics inferred from single-sample amplicon sequencing",
    author="Philippe Chlenski",
    author_email="pac@cs.columbia.edu",
    url="http://www.github.com/pchlenski/aptr",
    packages=["aptr"],
    package_dir={"aptr": "src"},
    install_requires=[
        "biopython",
        "numpy",
        "pandas",
        "uuid",
        "torch",
        "matplotlib",
    ],
)
