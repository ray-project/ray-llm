#!/bin/bash
set -euxo pipefail

GIT_COMMIT="${1:-}"

if [[ -n "$GIT_COMMIT" ]];then
    git diff-index --quiet HEAD --
    GIT_COMMIT=`git rev-parse HEAD`
fi

rm -rf dist
rm -rf build

python setup.py bdist_wheel