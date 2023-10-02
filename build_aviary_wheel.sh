#!/bin/bash
[[ "${DEBUG:-}" =~ [1..9]|[Tt]rue|[Yy]es ]] && set -x
set -euo pipefail

rm -rf dist
rm -rf build

python setup.py bdist_wheel
