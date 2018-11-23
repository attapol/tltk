# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tltk',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.1.6',

    description='Thai Language Toolkit',
    long_description=long_description,

    # The project's main homepage.
    url='http://pypi.python.org/pypi/tltk/',
    #url='https://github.com/tltk/',

    # Author details
    author='Wirote Aroonmanakun',
    author_email='awirote@gmail.com',

    # Choose your license
    license='GPLv3',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.4'
    ],

    # What does your project relate to?
    keywords='Thai language toolkit, Thai language processing, segmentation',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
#    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    packages=['tltk'],

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
     install_requires=[
          'nltk', 'sklearn', 'sklearn_crfsuite', 'gensim'
      ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
  #  extras_require={
  #      'dev': ['check-manifest'],
  #      'test': ['coverage'],
  #  },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'tltk': ['sylrule.lts', 'thaisyl.dict', 'PhSTrigram.sts', 'sylseg.3g', 'thdict', 'BEST.dict', 'tnc-tagger.pick', 'sent_segment_rfs.pick', 'ner-tagger.pick'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
#    data_files = [('corpus', ['tnc2gram', 'tnc3gram', 'tncwordlist'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'tltk = tltk:main'
        ],
    },
)
