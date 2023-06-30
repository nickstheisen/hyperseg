from setuptools import setup, find_packages

setup(
    name='Hyperspectral Semantic Segmentation',
    version='0.1.0',
    author='Nick Theisen',
    author_email='nicktheisen@uni-koblenz.de',
    packages=find_packages(),
    scripts=[],
    url='',
    license='LICENSE.txt',
    description='A framework for hyperspectral semantic segmentation experiments.',
    long_description=open('README.md').read(),
    install_requires=[
        "torch",
        "torchvision",
        "tensorboard",
        "h5py",
        "pytorch_lightning",
        "tqdm",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "imageio",
        "tifffile",
        "efficientnet_pytorch",
        "torchinfo"
        ],
)
