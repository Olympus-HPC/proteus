set -e

ml load python/3.9

echo "DIRS are under"
python -m venv $SYS_TYPE
source $SYS_TYPE/bin/activate
LLVM_INSTALL_DIR=${ROCM_PATH} PROTEUS_SRC=${CI_PROJECT_DIR} pip install -v git+https://github.com/Olympus-HPC/Mneme.git

