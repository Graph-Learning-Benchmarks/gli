#!/bin/bash

FILES=$(git ls-files '*.py')

set -Eeuo pipefail
set -x

pycodestyle $FILES
pydocstyle $FILES
pylint $FILES --rcfile .pylintrc
