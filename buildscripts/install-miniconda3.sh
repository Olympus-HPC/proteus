MACHINE=$(uname -m)
MINICONDA_DIR=miniconda3-${MACHINE}

function setup() {
  mkdir -p ./${MINICONDA_DIR}
  if [ -f ${MINICONDA_DIR}/bin/activate ]; then
    source ./${MINICONDA_DIR}/bin/activate
    return 0
  fi

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O ./${MINICONDA_DIR}/miniconda.sh
  bash ./${MINICONDA_DIR}/miniconda.sh -b -u -p ./${MINICONDA_DIR}
  rm ./${MINICONDA_DIR}/miniconda.sh
  source ./${MINICONDA_DIR}/bin/activate
}

setup