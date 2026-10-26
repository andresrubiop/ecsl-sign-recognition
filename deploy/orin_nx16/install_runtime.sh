#!/usr/bin/env bash
set -Eeuo pipefail

deploy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${deploy_dir}/../.." && pwd)"
venv_dir="${PFRUTAS_ORIN_VENV:-${project_root}/.venv-orin}"
venv_python="${venv_dir}/bin/python"

torch_jp61_url="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
cusparselt_name="libcusparse_lt-linux-sbsa-0.5.2.1-archive"
cusparselt_url="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/${cusparselt_name}.tar.xz"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ "$(uname -m)" != "aarch64" ]]; then
    fail "este instalador solo admite Linux aarch64 en Jetson Orin NX."
fi

ubuntu_version="$(. /etc/os-release; printf '%s' "${VERSION_ID:-desconocida}")"
if [[ "${ubuntu_version}" != "22.04" ]]; then
    fail "se requiere Ubuntu 22.04 de JetPack 6; se detectó ${ubuntu_version}."
fi

python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${python_version}" != "3.10" ]]; then
    fail "se requiere Python 3.10; se detectó ${python_version}."
fi

jetpack_version="$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null || true)"
if [[ -z "${jetpack_version}" ]]; then
    fail "no se encontró el metapaquete nvidia-jetpack; instala JetPack 6.1 o 6.2.x."
fi
case "${jetpack_version}" in
    6.1*) jetpack_line="6.1" ;;
    6.2*) jetpack_line="6.2" ;;
    *)
        fail "JetPack ${jetpack_version} no pertenece al perfil probado 6.1/6.2.x."
        ;;
esac

printf 'PFrutas Orin NX 16 GB\n'
printf '  proyecto: %s\n' "${project_root}"
printf '  entorno:  %s\n' "${venv_dir}"
printf '  JetPack:  %s\n' "${jetpack_version}"

sudo apt-get update
sudo apt-get install -y \
    curl \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-tools \
    libgl1 \
    libopenblas-dev \
    python3-matplotlib \
    python3-opencv \
    python3-pip \
    python3-venv \
    v4l-utils \
    xz-utils

if [[ -e "${venv_dir}/pyvenv.cfg" ]]; then
    if ! grep -Eq '^include-system-site-packages = true$' "${venv_dir}/pyvenv.cfg"; then
        fail "${venv_dir} ya existe sin --system-site-packages. Usa otra ruta en PFRUTAS_ORIN_VENV o recréalo manualmente."
    fi
else
    python3 -m venv --system-site-packages "${venv_dir}"
fi

"${venv_python}" -m pip install --upgrade "pip==25.1"
"${venv_python}" -m pip install --no-cache-dir \
    -r "${deploy_dir}/requirements-runtime.txt"

torch_cuda_ready() {
    "${venv_python}" -c \
        'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
        >/dev/null 2>&1
}

install_cusparselt_if_missing() {
    local cusparselt_tmp
    if /sbin/ldconfig -p 2>/dev/null | grep -q 'libcusparseLt\.so'; then
        printf 'cuSPARSELt ya está disponible; no se reemplazará.\n'
        return
    fi
    if [[ ! -d /usr/local/cuda/include || ! -d /usr/local/cuda/lib64 ]]; then
        fail "no se encontró /usr/local/cuda completo; revisa la instalación de JetPack/CUDA."
    fi

    cusparselt_tmp="$(mktemp -d -t pfrutas-cusparselt-XXXXXXXX)"
    printf 'Instalando cuSPARSELt requerido por el wheel NVIDIA...\n'
    curl --fail --location --retry 3 \
        --output "${cusparselt_tmp}/${cusparselt_name}.tar.xz" \
        "${cusparselt_url}"
    tar -C "${cusparselt_tmp}" -xf \
        "${cusparselt_tmp}/${cusparselt_name}.tar.xz"
    if [[ ! -d "${cusparselt_tmp}/${cusparselt_name}/include" \
        || ! -d "${cusparselt_tmp}/${cusparselt_name}/lib" ]]; then
        fail "el archivo cuSPARSELt descargado no contiene el layout esperado."
    fi
    sudo cp -a "${cusparselt_tmp}/${cusparselt_name}/include/." \
        /usr/local/cuda/include/
    sudo cp -a "${cusparselt_tmp}/${cusparselt_name}/lib/." \
        /usr/local/cuda/lib64/
    sudo /sbin/ldconfig
    rm -rf -- "${cusparselt_tmp}"
}

if torch_cuda_ready; then
    printf 'PyTorch CUDA ya está disponible; no se reemplazará.\n'
elif [[ "${jetpack_line}" == "6.1" ]]; then
    install_cusparselt_if_missing
    printf 'Instalando el wheel NVIDIA PyTorch 2.5 para JetPack 6.1...\n'
    "${venv_python}" -m pip install --no-cache-dir "${torch_jp61_url}"
    if ! torch_cuda_ready; then
        fail "el wheel NVIDIA se instaló, pero torch.cuda.is_available() sigue siendo False. Revisa JetPack/CUDA."
    fi
else
    fail "JetPack 6.2.x no tiene wheel NVIDIA independiente en la matriz oficial. Instala primero un PyTorch CUDA compatible y vuelve a ejecutar este script; no se instalará un wheel supuesto."
fi

"${venv_python}" -m pip install --no-cache-dir --no-deps --require-hashes \
    -r "${deploy_dir}/requirements-mediapipe-arm64.txt"
"${venv_python}" -m pip install --no-deps --editable "${project_root}"

"${venv_python}" -c '
import cv2
import mediapipe as mp
import torch

assert mp.__version__ == "0.10.18", mp.__version__
assert torch.cuda.is_available()
print(f"MediaPipe {mp.__version__}")
print(f"OpenCV {cv2.__version__}")
print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}")
print(f"GPU {torch.cuda.get_device_name(0)}")
'

"${venv_python}" "${project_root}/scripts/11_check_orin_nx16.py" \
    --bundle-dir "${project_root}/artifacts/orin_nx16_fusion" \
    --camera-mode usb \
    --skip-bundle

printf '\nInstalación terminada. Activa el entorno con:\n'
printf 'source %q\n' "${venv_dir}/bin/activate"
printf 'Después copia/construye el bundle y ejecuta el preflight completo.\n'
