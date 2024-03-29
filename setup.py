from setuptools import setup


setup_requires = []

install_requires = [
    "numpy>=1.16.6",
    "attrs",
    # https://github.com/HiroIshida/yamaopt/issues/18#issuecomment-1002915454
    "scipy>=1.2.0",
    'sympy==0.7;python_version<"3.0"',
    'sympy;python_version>="3.0"',
    "tinyfk>=0.6.0",
    "scikit-robot>=0.0.15",
]

setup(
    name='yamaopt',
    version='0.0.1',
    description='Optimizing the position where the robot attaches the sensor.',
    author='Hirokazu Ishida',
    author_email='h-ishida@jsk.imi.i.u-tokyo.ac.jp',
    url='https://github.com/HiroIshida/yamaopt',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    py_modules=[],
    install_requires=install_requires,
)
