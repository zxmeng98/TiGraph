from setuptools import setup, find_packages

def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


require_list = fetch_requirements("requirements.txt")

setup(
    name="od_execution",
    version="0.0.1",
    description="od_execution",
    author="yzs981130",
    author_email="yzs981130@126.com",
    url="https://yezhisheng.me/",
    install_requires=require_list,
    setup_requires=require_list,
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
