#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ANY2RSI_ANYCONTROL_ROOT:-${ROOT}/third_party/AnyControl}"
COMMIT="$(tr -d '[:space:]' < "${ROOT}/third_party/ANYCONTROL_COMMIT")"

if ! git -C "${DEST}" rev-parse --git-dir >/dev/null 2>&1; then
  if [[ "${DEST}" == "${ROOT}/third_party/AnyControl" && -f "${ROOT}/.gitmodules" ]]; then
    git -C "${ROOT}" submodule update --init --recursive third_party/AnyControl
  else
    git clone https://github.com/open-mmlab/AnyControl.git "${DEST}"
  fi
fi
if [[ "${DEST}" == "${ROOT}/third_party/AnyControl" ]]; then
  git -C "${ROOT}" config submodule.third_party/AnyControl.ignore untracked
fi
if ! git -C "${DEST}" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
  git -C "${DEST}" fetch origin "${COMMIT}"
fi
git -C "${DEST}" checkout --detach "${COMMIT}"

mkdir -p "${DEST}/annotator/ckpts"
for name in network-bsds500.pth CropFormer_hornet_3x_03823a.pth; do
  source_path="${ROOT}/weights/anycontrol/annotator/ckpts/${name}"
  target_path="${DEST}/annotator/ckpts/${name}"
  if [[ -f "${source_path}" && ! -e "${target_path}" ]]; then
    ln -s "${source_path}" "${target_path}"
  fi
done

echo "AnyControl ready at ${DEST} (${COMMIT})"
echo "Install requirements/annotators.txt and compile EntitySeg ops as described in docs/INSTALL.md."
