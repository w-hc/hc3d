from setuptools import setup, find_packages

setup(
    name='hc3d',
    version='0.1.0',
    author='Haochen Wang',
    author_email='whc@uchicago.edu',
    packages=find_packages(),
    python_requires='>=3',
    install_requires=[],
    package_data={
        # If any package contains *.yml, include them:
        '': ['*.yml'],
    },
    entry_points={
        'console_scripts': []
    },
    zip_safe=True  # accessing config files without using pkg_resources.
)
