#!/bin/bash

set -Eeuo pipefail
set -x

pycodestyle glb
pydocstyle glb
pylint glb --rcfile .pylintrc
