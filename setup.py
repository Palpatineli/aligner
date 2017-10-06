from setuptools import setup

setup(
    name='aligner',
    version='0.1',
    install_requires=["opencv_python>=3.0", 'libtiff', 'noformat',
                      'pytest', 'py-flags', 'scipy', 'numpy', 'numba', 'uifunc'],
    packages=['plptn', 'plptn.aligner'],
    namespace_packages=['plptn'],
    package_data={
        'example': ['*.py', '*.tif']
    },
    entry_points={
        'console_scripts': ['aligner_align = plptn.aligner.scripts.align:align_files',
                            'aligner_two_chan = plptn.aligner.scripts.align:align_two_chan',
                            'aligner_measure = plptn.aligner.scripts.apply_roi:apply_roi',
                            'aligner_rearrange = plptn.aligner.scripts.rearrange:rearrange_folder',]
    }
)
