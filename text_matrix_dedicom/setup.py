from setuptools import setup, find_packages

setup(
    name='text_dedicom',
    version='0.0.1',
    author='Lars Hillebrand and David biesner',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'matplotlib==3.2.1',
        'nltk==3.4.5',
        'numpy==1.18.1',
        'pandas==1.0.1',
        'pillow==8.2.0',
        'pyyaml==5.3',
        'requests==2.22.0',
        'scikit-learn==0.22.1',
        'scipy==1.4.1',
        'seaborn==0.10.0',
        'tensorboard==2.1.0',
        'torch==1.4.0',
        'torchvision==0.5.0',
        'tqdm==4.42.0',
        'umap-learn==0.3.10',
        'wikipedia-api==0.5.4'
    ]
)
