from setuptools import setup, find_packages

setup(
    name="webnlg_toolkit",
    version="0.0.2",
    author="Liam Cripwell, Yifei Song",
    author_email="liam.cripwell@loria.fr, yifei.song@loria.fr",
    description="A package to assist with WebNLG-related tasks.",
    packages=find_packages(),
    include_package_data=True,
)