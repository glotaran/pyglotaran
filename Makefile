all: install

install:
	python3 setup.py install

rebuild: uninstall clean install

uninstall:
	rm -rf /usr/lib64/python3.5/site-packages/glotaran*

clean:
	rm -rf build/

profile: profile_single_decay

profile_single_decay:
	# time python3 profiling/single_decay.py
	python3 /usr/lib64/python3.5/site-packages/kernprof.py -l -v profiling/single_decay.py

profile_single_decay_with_irf:
	# time python3 profiling/single_decay.py
	python3 /usr/lib64/python3.5/site-packages/kernprof.py -l -v profiling/single_decay_with_irf.py

profile_three_decay_with_irf:
	# time python3 profiling/single_decay.py
	python3 /usr/lib64/python3.5/site-packages/kernprof.py -l -v profiling/three_decay_with_irf.py
