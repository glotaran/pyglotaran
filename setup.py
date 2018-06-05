import setuptools
# This is just a placeholder setup.py to claim the glotaran name on PyPI
# It is not meant to be usuable in any way as of yet.
with open("README.md", "r") as fh:
    long_description = fh.read()
	
setuptools.setup(
    name="glotaran",	
    version="0.0.1",
    description='A placeholder for the python based glotaran package',    
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Joris Snellenburg',
    author_email='j.snellenburg@gmail.com',
	url='https://github.com/glotaran/glotaran',
	download_url = 'https://github.com/glotaran/glotaran/archive/0.0.1.tar.gz',
	keywords = ['placeholder','testing','not-ready-for-use'],
	packages=setuptools.find_packages(),
	classifiers=[
        'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
	],
)