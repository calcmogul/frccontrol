#!/bin/bash

rm -rf dist
git checkout main || exit 1

# Ensure no files are untracked, changed, or staged respectively
if [ `echo -n $(git clean -n) | wc -c` != 0 ]; then
  echo "Please remove untracked files before publishing (see 'git status')"
  exit 1
fi
if [ `echo -n $(git diff) | wc -c` != 0 ]; then
  echo "Please remove changed files before publishing (see 'git status')"
  exit 1
fi
if [ `echo -n $(git diff --staged) | wc -c` != 0 ]; then
  echo "Please remove staged files before publishing (see 'git status')"
  exit 1
fi

git pull https://github.com/calcmogul/frccontrol main || exit 1
python -m build
twine upload dist/*
