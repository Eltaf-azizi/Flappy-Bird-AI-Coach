from setuptools import setup, find_packages

setup(
    name='flappy-bird-ai-coach',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'pygame>=2.1',
        'numpy>=1.21',
        'torch>=1.12',
        'matplotlib',
        'opencv-python-headless',
        'gymnasium>=0.29',
        'tqdm'
    ],
)
