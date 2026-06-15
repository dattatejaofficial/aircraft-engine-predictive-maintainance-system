from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List:
    requirements_list : List[str] = []
    
    try:
        with open('requirements.txt','r') as file:
            lines = file.readlines()

            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirements_list.append(requirement)
    
    except FileNotFoundError:
        print('requirements.txt file not found')
    
    return requirements_list

setup(
    name='AircraftMaintainancePredictiveSystem',
    version='0.0.1',
    author='DATTA TEJA',
    author_email='dattateja1234@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)