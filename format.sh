#!/bin/bash
# Formats the files by running pre-commit hooks.

while getopts 'ah' opt; do
  case "$opt" in
    a)
      pip install -q pre-commit
      pre-commit install
      pre-commit run --all-files
      exit $?
      ;;

    ?|h)
      echo -e "Usage: $(basename $0) [-a]\n\t-a\trun on all files\n\t-h\tshow this help message"
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"

pip install -q pre-commit
pre-commit install
pre-commit run
exit $?