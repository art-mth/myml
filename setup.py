import setuptools

setuptools.setup(
    name='myml',
    version='0.1',
    description='My machine learning library',
    author='Arthur Mathies',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ])
