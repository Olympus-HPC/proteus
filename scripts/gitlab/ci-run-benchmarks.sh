#!/bin/bash

set -e

echo "CI_COMMIT_REF_NAME ${CI_COMMIT_REF_NAME}"
# Fetch the PR ID from the branch name.
PR_INFO=$(curl -s -L -H "Authorization: Bearer $GITHUB_TOKEN" \
               -H "Accept: application/vnd.github+json" \
               -H "X-GitHub-Api-Version: 2022-11-28" \
               "https://api.github.com/repos/Olympus-HPC/proteus/pulls?head=Olympus-HPC:${CI_COMMIT_REF_NAME}")

# Check if PR exists.
if [ -z "${PR_INFO}" ] || [ "$(echo "$PR_INFO" | jq length)" = "0" ]; then
  echo "No PR found for ref ${CI_COMMIT_REF_NAME}, exit"
  exit 0
fi

# Extract PR number.
PR_ID=$(echo "${PR_INFO}" | jq -r '.[0].number')
echo "Processing PR ${PR_ID}"

COMMENTS_INFO=$(curl -L \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/Olympus-HPC/proteus/issues/${PR_ID}/comments")
COMMENTS_BODY=$(echo ${COMMENTS_INFO} | jq -r '.[].body')
if [[ "${COMMENTS_BODY}" == *"/run-benchmarks-hecbench"* ]]; then
  echo "=> Run hecbench benchmarks triggered <=";
  BENCHMARKS_TOML="hecbench.toml"
  REPS="1"
elif [[ "${COMMENTS_BODY}" == *"/run-benchmarks-rajaperf"* ]]; then
  echo "=> Run rajaperf benchmarks triggered <=";
  BENCHMARKS_TOML="rajaperf.toml"
  REPS="1"
elif [[ "${COMMENTS_BODY}" == *"/run-benchmarks-lbann"* ]]; then
  echo "=> Run lbann benchmarks triggered <=";
  BENCHMARKS_TOML="lbann.toml"
  REPS="5"
else
  echo "=> Benchmarks will not run, trigger with /run-benchmarks-{hecbench|rajaperf|lbann} <="
  exit 0
fi

echo "Activate proteus environment..."
source /usr/workspace/proteusdev/${CI_MACHINE}/miniconda3/bin/activate
conda activate proteus

if [ "${CI_MACHINE}" == "lassen" ]; then
  if [ "${BENCHMARKS_TOML}" == "rajaperf.toml" ]; then
    echo "RAJAPerf benchmarks can only run on tioga.  Exiting."
    exit 0
  fi
  if [ "${BENCHMARKS_TOML}" == "lbann.toml" ]; then
    echo "LBANN benchmarks can only run on tioga.  Exiting."
    exit 0
  fi
  ml load cuda/12.2.2

  LLVM_INSTALL_DIR=$(llvm-config --prefix)

  CMAKE_MACHINE_OPTIONS="\
    -DPROTEUS_LINK_SHARED_LLVM=on \
    -DPROTEUS_ENABLE_CUDA=on \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_CUDA_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
  "
  PROTEUS_CC=${CONDA_PREFIX}/bin/clang++
  MACHINE=nvidia
elif [ "${CI_MACHINE}" == "tioga" ]; then
  ml load rocm/6.2.1

  LLVM_INSTALL_DIR=${ROCM_PATH}/llvm

  if [ "${BENCHMARKS_TOML}" == "lbann.toml" ]; then
    ml load cpe
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CRAY_LD_LIBRARY_PATH
    CMAKE_MACHINE_OPTIONS="\
      -DPROTEUS_ENABLE_HIP=on \
      -DCMAKE_POSITION_INDEPENDENT_CODE=on \
    "
  else
    CMAKE_MACHINE_OPTIONS="\
      -DPROTEUS_ENABLE_HIP=on \
    "
  fi

  PROTEUS_CC=hipcc
  MACHINE=amd
else
  echo "Unsupported machine ${CI_MACHINE}"
  exit 1
fi

echo "Build proteus..."
mkdir build
pushd build

PROTEUS_INSTALL_PATH=${PWD}/install
CMAKE_OPTIONS="\
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR} \
  -DCMAKE_INSTALL_PREFIX=${PROTEUS_INSTALL_PATH} \
  -DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
  -DCMAKE_CXX_COMPILER=${LLVM_INSTALL_DIR}/bin/clang++ \
  -DENABLE_TESTS=off \
"

CMAKE_OPTIONS+=${CMAKE_MACHINE_OPTIONS}
cmake ${CI_PROJECT_DIR} ${CMAKE_OPTIONS}
make -j install
popd

# Run under the project directory to avoid deleting artifacts in the
# after_script.
cd ${CI_PROJECT_DIR}
git clone --depth 1 --recursive --single-branch --branch proteus-ci-testing https://github.com/Olympus-HPC/proteus-benchmarks.git

cd proteus-benchmarks

echo "Running AOT"
python driver.py -t ${BENCHMARKS_TOML} \
  -c ${PROTEUS_CC} -j ${PROTEUS_INSTALL_PATH} -x aot -m ${MACHINE} -r ${REPS} --runconfig presets/aot.toml --results-dir results
echo "Running proteus"
python driver.py -t ${BENCHMARKS_TOML} \
  -c ${PROTEUS_CC} -j ${PROTEUS_INSTALL_PATH} -x proteus \
  -m ${MACHINE} -r ${REPS} --runconfig presets/proteus.toml --results-dir results

echo "End run benchmarks..."

python vis/plot-bar-end2end-speedup.py --dir results --plot-dir plots -m ${MACHINE} -f png --plot-title ${CI_MACHINE}
python vis/plot-bar-end2end-speedup-noopt.py --dir results --plot-dir plots -m ${MACHINE} -f png --plot-title ${CI_MACHINE}
python vis/plot-bar-compilation-slowdown.py  --dir results --plot-dir plots -m ${MACHINE} -f png --plot-title ${CI_MACHINE}
python vis/plot-bar-kernel-speedup-ablation.py --dir results --plot-dir plots -m ${MACHINE} -f png --plot-title ${CI_MACHINE}

# Upload artifact to github.
echo "Uploading artifacts to github..."

# Prepare the comment body with artifact info.
RESULTS_COMMIT="Artifacts PR ${PR_ID} commit ${CI_COMMIT_SHORT_SHA}"
COMMENT="${RESULTS_COMMIT}\n<p>"

RESULTS_REPO=github.com/Olympus-HPC/proteus-benchmark-results
git clone https://$GITHUB_TOKEN@${RESULTS_REPO}
OUTPUT_DIR="proteus-benchmark-results"
ARTIFACT_DIR="PR${PR_ID}/${CI_COMMIT_SHORT_SHA}"
mkdir -p ${OUTPUT_DIR}/${ARTIFACT_DIR}
# Loop through each artifact file.
for file in plots/*.png; do
  FILENAME=$(basename "${file}")
  cp ${file} ${OUTPUT_DIR}/${ARTIFACT_DIR}
  # Add to the comment body.
  COMMENT+="\n<img src=https://${RESULTS_REPO}/blob/main/${ARTIFACT_DIR}/${FILENAME}?raw=true width=49%>"
done
COMMENT+="\n</p>"
cd ${OUTPUT_DIR}
git add PR${PR_ID}
git commit -m "${RESULTS_COMMIT}"
git push

# Post the comment to the GitHub PR.
curl -L -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/Olympus-HPC/proteus/issues/${PR_ID}/comments" \
  -d "{\"body\": \"${COMMENT}\"}"
