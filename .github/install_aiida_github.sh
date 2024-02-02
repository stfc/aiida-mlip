#!/bin/bash
git clone https://github.com/aiidateam/aiida_core ../aiida_core
cd ../aiida_core
git checkout $AIIDA_DEVELOP_GIT_HASH
pip install -e .[docs,pre-commit,testing]
cd ${TRAVIS_BUILD_DIR}
