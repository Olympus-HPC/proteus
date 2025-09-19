set -e

ml load python/3.9

python -m venv $SYS_TYPE
source $SYS_TYPE/bin/activate
PROTEUS_SRC=${CI_PROJECT_DIR} pip install https://github.com/Olympus-HPC/Mneme.git

