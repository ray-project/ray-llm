#!/bin/bash
set -euxo pipefail

git diff-index --quiet HEAD --
GIT_COMMIT=`git rev-parse HEAD`

rm -rf dist
rm -rf build

python setup.py bdist_wheel