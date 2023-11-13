# Signifies our desired python version
PYTHON = python
PYTHON_FILES := gli/ benchmarks/ tests/ example.py

# .PHONY defines parts of the makefile that are not dependant on any specific file
# This is most often used to store functions
.PHONY = help setup test run clean

# The @ makes sure that the command itself isn't echoed in the terminal
help:
	@echo "Usage: make <command>"
	@echo " Available commands:"
	@echo "  setup:    install the full project."
	@echo "  clean:    remove all data files (npz)."
	@echo "  test:     run all tests (pystyle, pylint, pytest). Stop on failure."
	@echo "  pystyle:  run pycodestyle and pydocstyle tests."
	@echo "  pylint:   run pylint."
	@echo "  pytest:   run pytests on all datasets by default."
	@echo "            run pytest on a single dataset by arg DATASET."
	@echo "            e.g., make pytest DATASET=cora"
	@echo "  donwload: download and preprocess all data files (npz)."
	@echo "  logs:     create logs directory."
	@echo "  build:    build the package."

setup:
	${PYTHON} -m pip install -e ".[test,full]"

clean:
	find datasets -name '*.npz' -delete

test: pystyle pylint pytest

# The leading '-' will execute following command to ignore errors.
pystyle: logs
	-pycodestyle ${PYTHON_FILES} | tee logs/pycodestyle.log
	-pydocstyle ${PYTHON_FILES} | tee logs/pydocstyle.log

pylint: logs
	-pylint ${PYTHON_FILES} --rcfile .pylintrc --recursive y | tee logs/pylint.log

pytest: logs
ifndef DATASET
	-pytest -v tests/ | tee logs/pytest.log
else
	mkdir -p temp
	@echo $$DATASET > temp/changed_datasets
	-pytest -v tests/ | tee logs/pytest.log
	rm temp/changed_datasets
	rmdir temp
endif

download:
	${PYTHON} tests/preprocess.py

logs:
	-mkdir logs

build:
	${PYTHON} -m build