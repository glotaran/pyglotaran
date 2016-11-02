all: install

install:
	python3 setup.py install

rebuild: uninstall clean install

uninstall: 
	rm -rf /usr/lib64/python3.5/site-packages/glotaran_*

clean:
	rm -rf build/
