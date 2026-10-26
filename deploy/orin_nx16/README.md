# Despliegue en Jetson Orin NX 16 GB

Este perfil ejecuta el modelo ya entrenado; no prepara `raw` ni vuelve a entrenar.
El flujo de cámara permanece fiel al experimento:

```text
USB o CSI -> OpenCV/GStreamer -> MediaPipe Holistic (CPU)
          -> features BiLSTM + grafo ST-GCN -> PyTorch CUDA/FP16 -> interfaz
```

El perfil `full` carga los cinco folds externos y el refit final: seis miembros y
aproximadamente 75 MiB de pesos portables. El perfil `final` carga un solo miembro
para ahorrar memoria y latencia, pero no hereda la métrica OOF del ensemble.

## Plataforma soportada

El objetivo reproducible principal es:

- Jetson Orin NX 16 GB;
- JetPack 6.1, Ubuntu 22.04 y Python 3.10;
- wheel PyTorch 2.5 CUDA oficial de NVIDIA para JetPack 6.1;
- MediaPipe 0.10.18, wheel oficial PyPI para CPython 3.10/Linux ARM64;
- OpenCV instalado por JetPack/Ubuntu, conservando GStreamer.

JetPack 6.2.x también se admite si `torch.cuda.is_available()` ya devuelve `True`.
La matriz oficial muestra containers para 6.2, pero no un wheel NVIDIA independiente;
por eso el instalador no construye una URL supuesta ni reemplaza un PyTorch funcional.
JetPack 7 no forma parte de este perfil.

MediaPipe 0.10.21, usado para preparar el dataset, no tiene wheel Linux ARM64. En
0.10.18 se comprobaron byte a byte `holistic.py`, `solution_base.py`, el grafo
`holistic_landmark_cpu.binarypb` y todos los modelos TFLite usados por Holistic:
coinciden con 0.10.21. El preflight valida además el contrato instalado. El wheel
MediaPipe procesa landmarks en CPU; CUDA/FP16 se usa en el clasificador PyTorch.

Referencias primarias:

- [JetPack 6.1 y soporte Orin](https://docs.nvidia.com/jetson/jetpack/6.1/release-notes/index.html)
- [Instalación oficial de PyTorch en Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [Matriz PyTorch/JetPack](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)
- [MediaPipe 0.10.18 ARM64 en PyPI](https://pypi.org/project/mediapipe/0.10.18/)

## Compatibilidad experimental: JetPack 5.1.5 / Ubuntu 20.04

También hay un perfil separado para una Orin NX 16 GB que deba permanecer en
Ubuntu 20.04. No es una instalación sobre Ubuntu genérico: se debe flashear el
BSP **JetPack 5.1.5 / L4T 35.6.1** de NVIDIA para la carrier board concreta.
Ese BSP soporta Orin NX, usa Ubuntu 20.04 y proporciona CUDA 11.4.19, TensorRT
8.5.2 y cuDNN 8.6.0. Consulta la [nota de JetPack 5.1.5](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-515/release-notes/index.html)
y la [guía de flasheo de L4T 35.6.1](https://docs.nvidia.com/jetson/archives/r35.6.1/DeveloperGuide/SD/FlashingSupport.html)
antes de reflashear; el procedimiento depende de la carrier y puede borrar el
disco de la placa.

El perfil utiliza Python 3.8, el wheel CUDA NVIDIA de PyTorch 2.1 para JetPack
5.1.x y MediaPipe 0.10.5 CPython 3.8/ARM64, fijado por SHA-256. El último se
elige porque MediaPipe 0.10.18 no publicó un wheel ARM64 para CPython 3.8. Los
activos de Holistic relevantes coinciden con el extractor del experimento salvo
`solution_base.py`; el preflight comprueba una huella específica para esa versión.
El mismo bundle `artifacts/orin_nx16_fusion` se reutiliza: no se exportan pesos
nuevos ni se reentrena el modelo.

En la placa ya flasheada con JetPack 5.1.5:

```bash
cd /home/usuario/Pfrutas
bash deploy/orin_nx16/install_runtime_jp515_ubuntu2004.sh
source .venv-orin-jp5/bin/activate

python scripts/11_check_orin_nx16.py \
  --runtime-profile jetpack5 \
  --bundle-dir artifacts/orin_nx16_fusion \
  --camera-mode usb \
  --json-out artifacts/orin_nx16_fusion/preflight_jp5_usb.json
```

El instalador detiene la ejecución si no detecta ARM64, Ubuntu 20.04, Python 3.8,
`nvidia-jetpack` 5.1.5, L4T R35.6.1, el PyTorch CUDA esperado o MediaPipe 0.10.5.
Con el preflight aprobado, los comandos de las secciones 4 y 5 son los mismos;
solo conserva `--runtime-profile jetpack5` al repetir el preflight. Ejecuta primero
el benchmark `full` y comprueba en su JSON que la paridad golden tiene
`argmax_matches: true` y `within_tolerance: true` antes de usar cámara.

Este perfil es **experimental hasta validarlo en una Orin física** con el preflight,
benchmark CUDA y cámara USB/CSI. La compatibilidad Python 3.8 se limita a la ruta
de inferencia; el paquete de desarrollo conserva su requisito Python 3.10+ para
entrenamiento. Ubuntu 20.04 terminó su soporte estándar, de modo que una placa
conectada debería usar Ubuntu Pro/ESM o una red gestionada. Referencias del wheel:
[PyTorch NVIDIA para Jetson](https://developer.nvidia.com/w/compute/redist/jp/v512/pytorch/)
y [MediaPipe 0.10.5 ARM64](https://pypi.org/project/mediapipe/0.10.5/).

## 1. Crear el bundle en el computador de desarrollo

Desde la raíz de PFrutas, donde ya existe la corrida entrenada:

```bash
source .venv/bin/activate
python scripts/08_build_orin_nx16_bundle.py \
  --run-dir artifacts/fusion_asl_citizen_stgcn_v1_gpu \
  --output-dir artifacts/orin_nx16_fusion \
  --parity-sample \
    data/processed_asl_transfer_v1/sequences/badea__badea1__3459c353.npz \
  --parity-device cpu
```

El builder genera checkpoints portables que pueden cargarse con
`torch.load(weights_only=True)`, conserva las métricas OOF y escribe un manifiesto
con SHA-256. No necesitas copiar `raw`, `data/processed_*`, el checkpoint ASL original
ni los historiales de entrenamiento.

Copia al Jetson el repositorio y `artifacts/orin_nx16_fusion`. Por ejemplo, desde el
computador de desarrollo:

```bash
rsync -av --exclude .venv --exclude raw --exclude data --exclude artifacts \
  ./ usuario@jetson:/home/usuario/Pfrutas/
rsync -av artifacts/orin_nx16_fusion/ \
  usuario@jetson:/home/usuario/Pfrutas/artifacts/orin_nx16_fusion/
```

## 2. Instalar el runtime en el Jetson

En el Jetson:

```bash
cd /home/usuario/Pfrutas
bash deploy/orin_nx16/install_runtime.sh
source .venv-orin/bin/activate
```

El instalador:

- exige ARM64, Ubuntu 22.04, Python 3.10 y JetPack 6.1/6.2.x;
- crea `.venv-orin` con acceso al OpenCV del sistema;
- conserva cualquier PyTorch CUDA funcional;
- en JetPack 6.1 instala cuSPARSELt si falta y luego el wheel oficial NVIDIA;
- instala MediaPipe 0.10.18 con SHA-256 y `--no-deps`;
- instala PFrutas con `--no-deps`, evitando que pip sustituya OpenCV o PyTorch.

No instales `opencv-python` ni `opencv-contrib-python` con pip en este entorno: esos
wheels pueden ocultar el OpenCV de JetPack y perder `nvarguscamerasrc`/GStreamer.
Para colocar el entorno en otra ruta:

```bash
PFRUTAS_ORIN_VENV=/home/usuario/pfrutas-venv \
  bash deploy/orin_nx16/install_runtime.sh
```

## 3. Potencia y preflight

El instalador no cambia frecuencia, potencia ni ventilación. Consulta primero:

```bash
sudo nvpmodel -q
```

Para sesiones sostenidas se recomienda el modo de 25 W con refrigeración activa.
En los perfiles oficiales normal y Super del Orin NX 16 GB corresponde al ID 3:

```bash
sudo nvpmodel -m 3
```

El cambio puede solicitar reinicio. No uses `-m 0` a ciegas: según cómo se flasheó
la placa significa MAXN o MAXN_SUPER, ambos experimentales. La tabla oficial está en
[Power and Performance](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html).

Valida hardware, CUDA, una operación GPU real, MediaPipe, OpenCV y todos los hashes
del bundle antes de abrir la cámara:

```bash
python scripts/11_check_orin_nx16.py \
  --bundle-dir artifacts/orin_nx16_fusion \
  --camera-mode usb \
  --json-out artifacts/orin_nx16_fusion/preflight_usb.json
```

Para CSI cambia `--camera-mode csi`; en ese modo OpenCV con GStreamer es obligatorio.

## 4. Inferencia en vivo

Cámara USB `/dev/video0`, ensemble completo:

```bash
python scripts/09_live_inference_orin_nx16.py \
  --run-dir artifacts/orin_nx16_fusion \
  --profile full \
  --camera 0
```

Solo una fruta, ocultando toda información de las demás:

```bash
python scripts/09_live_inference_orin_nx16.py \
  --run-dir artifacts/orin_nx16_fusion \
  --profile full \
  --fruit MANGO \
  --camera 0
```

Cámara CSI mediante `nvarguscamerasrc`:

```bash
python scripts/09_live_inference_orin_nx16.py \
  --run-dir artifacts/orin_nx16_fusion \
  --profile full \
  --csi \
  --sensor-id 0 \
  --width 1280 \
  --height 720 \
  --camera-fps 30 \
  --prediction-fps 5
```

Para priorizar latencia usa un solo refit:

```bash
python scripts/09_live_inference_orin_nx16.py \
  --run-dir artifacts/orin_nx16_fusion \
  --profile final \
  --camera 0
```

Empieza con `--prediction-fps 5`: limita el ensemble, pero MediaPipe conserva el
muestreo temporal de 30 FPS del entrenamiento. Si MediaPipe satura CPU, reduce la
cámara a 960x540; bajar solo `--prediction-fps` no reduce ese extractor. El launcher
exige CUDA y checkpoints portables seguros. La opción
`--allow-unsafe-checkpoint-loading` existe solo para artefactos legados confiables y
no se necesita con el bundle generado.

Controles: `Espacio` registra el intento, `R` reinicia la sesión y `Q` o `Esc` sale.

## 5. Medir antes de elegir perfil

Ejecuta el benchmark sin cámara para separar el costo del clasificador del costo de
MediaPipe/OpenCV:

```bash
python scripts/10_benchmark_orin_nx16.py \
  --bundle-dir artifacts/orin_nx16_fusion \
  --mode full \
  --output artifacts/orin_nx16_fusion/benchmark_full.json

python scripts/10_benchmark_orin_nx16.py \
  --bundle-dir artifacts/orin_nx16_fusion \
  --mode final \
  --output artifacts/orin_nx16_fusion/benchmark_final.json
```

Conserva el JSON del benchmark junto con el reporte de preflight y anota modo de
potencia, cámara, resolución y temperatura. La exactitud OOF de **81,94 %** pertenece
al ensemble completo sobre 72 videos con LOSO por signante. No garantiza 81,94 % en
una cámara nueva; iluminación, encuadre, distancia y personas nuevas deben evaluarse
en el Jetson. Tampoco se debe atribuir esa cifra al perfil `final` sin una evaluación
separada.

Para USB, inspecciona formatos reales si la cámara no abre:

```bash
v4l2-ctl --list-devices
v4l2-ctl --device /dev/video0 --list-formats-ext
```

Para CSI, comprueba primero que Argus ve el sensor y que el preflight informa
`GStreamer: YES`. Un error de cámara no implica que el modelo o CUDA estén dañados.
