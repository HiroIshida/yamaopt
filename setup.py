from setuptools import setup

setup_requires = []

install_requires = [
        "numpy>=1.16.6",
        "attrs",
        # https://github.com/HiroIshida/yamaopt/issues/18#issuecomment-1002915454
        "scipy>=1.2.0",
        "tinyfk>=0.4.4",
        "scikit-robot>=0.0.15",
        ]

setup(
    name='yamaopt',
    version='0.0.1',
    description='',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    install_requires=install_requires,
)
