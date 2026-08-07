#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${repo_dir}/build/native-flexiv-rt"
rdk_source="${FLEXIV_RDK_SOURCE_DIR:-}"

cmake_args=(
  -S "${repo_dir}/native/flexiv_rt_controller"
  -B "${build_dir}"
  -DCMAKE_BUILD_TYPE=Release
  "-DFLEXIV_RDK_SOURCE_DIR=${rdk_source}"
)
if [[ -x "${repo_dir}/.venv/bin/python" ]]; then
  cmake_args+=("-DPython3_EXECUTABLE=${repo_dir}/.venv/bin/python")
fi
cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --parallel
echo "${build_dir}/dfc-flexiv-rt-controller"
