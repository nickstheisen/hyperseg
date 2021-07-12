from setuptools import setup

setup(
    name='Hyperspectral Semantic Segmentation',
    version='0.1.0',
    author='Nick Theisen',
    author_email='nicktheisen@uni-koblenz.de',
    packages=['hyperseg'],
    scripts=[],
    url='',
    license='LICENSE.txt',
    description='A framework for hyperspectral semantic segmentation experiments.',
    long_description=open('README.md').read(),
    install_requires=[
        "torch"
        ],
)
