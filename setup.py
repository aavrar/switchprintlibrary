#!/usr/bin/env python3
"""Setup script for SwitchPrint - Multilingual Code-Switching Detection Library."""

from setuptools import setup, find_packages
import os

# Read version from the package
def get_version():
    """Get version from the package."""
    here = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(here, 'codeswitch_ai', 'utils', 'constants.py')
    
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('VERSION'):
                return line.split('=')[1].strip().strip('"\'')
    
    return "0.1.0"

# Read README for long description
def get_long_description():
    """Get long description from README."""
    here = os.path.abspath(os.path.dirname(__file__))
    readme_file = os.path.join(here, 'README.md')
    
    try:
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "A comprehensive Python library for detecting, analyzing, and remembering multilingual code-switching patterns."

# Core requirements
CORE_REQUIREMENTS = [
    'langdetect==1.0.9',
    'sentence-transformers>=2.7.0',
    'faiss-cpu>=1.8.0',
    'numpy>=1.24.0',
    'scikit-learn>=1.3.0',
]

# Optional requirements
OPTIONAL_REQUIREMENTS = {
    'full': [
        'pandas>=2.0.0',
        'transformers>=4.39.0',
        'torch>=2.2.0',
    ],
    'ui': [
        'streamlit>=1.32.0',
        'flask>=2.3.0',
        'click>=8.1.0',
        'plotly>=5.17.0',
    ],
    'dev': [
        'pytest>=7.0.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
    ]
}

# All optional requirements combined
OPTIONAL_REQUIREMENTS['all'] = (
    OPTIONAL_REQUIREMENTS['full'] + 
    OPTIONAL_REQUIREMENTS['ui'] + 
    OPTIONAL_REQUIREMENTS['dev']
)

setup(
    name='switchprint',
    version=get_version(),
    author='Aahad Vakani',
    author_email='contact@aahadvakani.com',
    description='A state-of-the-art Python library for detecting, analyzing, and remembering multilingual code-switching patterns in text',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/aahadvakani/switchprint',
    project_urls={
        'Bug Reports': 'https://github.com/aahadvakani/switchprint/issues',
        'Source': 'https://github.com/aahadvakani/switchprint',
        'Documentation': 'https://github.com/aahadvakani/switchprint#readme',
    },
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='nlp, multilingual, code-switching, language-detection, ai, linguistics',
    python_requires='>=3.8',
    install_requires=CORE_REQUIREMENTS,
    extras_require=OPTIONAL_REQUIREMENTS,
    entry_points={
        'console_scripts': [
            'switchprint=codeswitch_ai.interface.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'codeswitch_ai': [
            'utils/*.py',
            'data/*.json',
        ],
    },
    zip_safe=False,
    license='MIT',
)