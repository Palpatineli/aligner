from setuptools import setup

setup(
    name='aligner',
    version='0.1',
    install_requires=["opencv_python>=3.0", 'tiffcapture', 'libtiff', 'pandas', "noformat", 'tqdm',
                      'pytest', 'py-flags', 'scipy'],
    packages=['plptn', 'plptn.aligner'],
    namespace_packages=['plptn'],
    package_data={
        'example': ['*.py', '*.tif']
    }
)
