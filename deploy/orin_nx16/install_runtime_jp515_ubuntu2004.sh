#!/usr/bin/env bash
set -Eeuo pipefail

# Perfil de compatibilidad experimental para Orin NX 16 GB con JetPack 5.1.5
# (L4T 35.6.1 / Ubuntu 20.04 / Python 3.8). No usar en Ubuntu genérico: CUDA,
# drivers, OpenCV y GStreamer deben provenir del BSP de NVIDIA JetPack.

deploy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${deploy_dir}/../.." && pwd)"
venv_dir="${PFRUTAS_ORIN_JP5_VENV:-${project_root}/.venv-orin-jp5}"
venv_python="${venv_dir}/bin/python"

# NVIDIA publica este wheel para la familia JetPack 5.1.x. Se valida versión y
# una operación CUDA real; no se sustituye por un torch genérico de PyPI.
torch_jp5_url="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0%2B41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
torch_expected_version="2.1.0a0+41361538.nv23.06"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ "$(uname -m)" != "aarch64" ]]; then
    fail "este instalador solo admite Linux aarch64 en Jetson Orin NX."
fi

ubuntu_version="$(. /etc/os-release; printf '%s' "${VERSION_ID:-desconocida}")"
if [[ "${ubuntu_version}" != "20.04" ]]; then
    fail "se requiere Ubuntu 20.04 del BSP JetPack 5; se detectó ${ubuntu_version}."
fi

python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${python_version}" != "3.8" ]]; then
    fail "se requiere Python 3.8 de JetPack 5; se detectó ${python_version}."
fi

jetpack_version="$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null || true)"
if [[ "${jetpack_version}" != 5.1.5* ]]; then
    fail "se requiere nvidia-jetpack 5.1.5.x; se detectó ${jetpack_version:-ausente}."
fi

l4t_release="$(tr -d '\000' < /etc/nv_tegra_release 2>/dev/null || true)"
if [[ "${l4t_release}" != *"R35 (release), REVISION: 6.1"* ]]; then
    fail "se requiere L4T R35.6.1 de JetPack 5.1.5; se detectó ${l4t_release:-ausente}."
fi

printf 'PFrutas Orin NX 16 GB — perfil experimental JP5/U20\n'
printf '  proyecto: %s\n' "${project_root}"
printf '  entorno:  %s\n' "${venv_dir}"
printf '  JetPack:  %s\n' "${jetpack_version}"

sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-tools \
    libgl1 \
    libopenblas-dev \
    python3-dev \
    python3-matplotlib \
    python3-opencv \
    python3-pip \
    python3-venv \
    v4l-utils

if [[ -e "${venv_dir}/pyvenv.cfg" ]]; then
    if ! grep -Eq '^include-system-site-packages = true$' "${venv_dir}/pyvenv.cfg"; then
        fail "${venv_dir} ya existe sin --system-site-packages. Usa otra ruta en PFRUTAS_ORIN_JP5_VENV o recréalo manualmente."
    fi
else
    python3 -m venv --system-site-packages "${venv_dir}"
fi

# pip 25 elimina Python 3.8. Se fija un conjunto que todavía lo soporta para
# poder instalar el proyecto editable sin resolver sus dependencias de training.
"${venv_python}" -m pip install --upgrade \
    "pip==24.3.1" \
    "setuptools==75.3.2" \
    "wheel==0.45.1"
"${venv_python}" -m pip install --no-cache-dir \
    -r "${deploy_dir}/requirements-runtime-jp5-py38.txt"

torch_is_expected() {
    "${venv_python}" -c \
        "import torch; raise SystemExit(0 if str(torch.__version__) == '${torch_expected_version}' and torch.cuda.is_available() else 1)" \
        >/dev/null 2>&1
}

if "${venv_python}" -c 'import torch' >/dev/null 2>&1; then
    if ! torch_is_expected; then
        fail "${venv_dir} contiene un PyTorch CUDA distinto de ${torch_expected_version}. Usa otra ruta PFRUTAS_ORIN_JP5_VENV para no mezclar perfiles."
    fi
    printf 'PyTorch CUDA compatible ya está disponible; no se reemplazará.\n'
else
    printf 'Instalando el wheel NVIDIA PyTorch %s para JetPack 5.1.x...\n' "${torch_expected_version}"
    "${venv_python}" -m pip install --no-cache-dir "${torch_jp5_url}"
    if ! torch_is_expected; then
        fail "el wheel NVIDIA se instaló, pero versión o CUDA no coinciden. Revisa JetPack/CUDA."
    fi
fi

"${venv_python}" -m pip install --no-cache-dir --no-deps --require-hashes \
    -r "${deploy_dir}/requirements-mediapipe-arm64-jp5-py38.txt"

# La metadata de desarrollo exige Python >=3.10 por el flujo completo de
# entrenamiento. El código de inferencia se mantiene compatible con 3.8 y se
# instala sin resolver esas dependencias modernas de desarrollo.
"${venv_python}" -m pip install --no-deps --no-build-isolation \
    --ignore-requires-python --editable "${project_root}"

"${venv_python}" -c '
import cv2
import mediapipe as mp
import torch

assert mp.__version__ == "0.10.5", mp.__version__
assert torch.__version__ == "2.1.0a0+41361538.nv23.06", torch.__version__
assert torch.cuda.is_available()
print(f"MediaPipe {mp.__version__}")
print(f"OpenCV {cv2.__version__}")
print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}")
print(f"GPU {torch.cuda.get_device_name(0)}")
'

"${venv_python}" "${project_root}/scripts/11_check_orin_nx16.py" \
    --runtime-profile jetpack5 \
    --bundle-dir "${project_root}/artifacts/orin_nx16_fusion" \
    --camera-mode usb \
    --skip-bundle

printf '\nInstalación terminada. Activa el entorno con:\n'
printf 'source %q\n' "${venv_dir}/bin/activate"
printf 'Después copia/construye el bundle y ejecuta el preflight completo con --runtime-profile jetpack5.\n'
