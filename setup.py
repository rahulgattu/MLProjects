from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(filepath: str)->List[str]:
    """This function returns the list of requirements

    Args:
        filepath (str): Path of requirements.txt 

    Returns:
        List[str]: requirements list
    """
    
    requirements = list()
    with open(filepath) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.strip() for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements
        
    
     


setup(
    name='ML_Project',
    version='0.0.1',
    author='Rahul Gattu',
    author_email='rahulgattu1998@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
    
) 