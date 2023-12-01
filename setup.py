from setuptools import setup, find_packages

setup(
    name="webnlg_toolkit",
    version="0.0.1",
    author="Liam Cripwell",
    author_email="liam.cripwell@loria.fr",
    description="A package to assist with WebNLG-related tasks.",
    packages=find_packages(),
    include_package_data=True,
)