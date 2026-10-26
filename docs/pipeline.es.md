# PFrutas: reconocimiento de señas de frutas

> Edición pública de código: no incluye vídeos, landmarks, pesos ni artefactos de entrenamiento. Las rutas de artefactos de esta guía describen ejecuciones locales; consulta [MODEL_RELEASE.md](../MODEL_RELEASE.md) para el alcance de distribución y [README.md](../README.md) para instalación.

PFrutas clasifica 16 señas de frutas a partir de video. El flujo toma la idea
 —landmarks de MediaPipe y clasificación temporal con LSTM— y la
adapta al dataset pequeño con validación por signante, features robustas y una
BiLSTM compacta con atención. La mejora 4 añade, como experimento separado, un
encoder ST-GCN preentrenado con ASL Citizen y lo fusiona con esa rama local.

La versión actual es el pipeline v2:

- 76 videos raw, 5 signantes y 16 clases;
- 72 videos incluidos por defecto y 4 tomas `_x` excluidas;
- landmarks de pose, manos y cara extraídos a 30 FPS;
- máscaras de presencia, interpolación limitada y deltas sin “manos fantasma”;
- segmentación automática del gesto y normalización temporal por fase;
- aumentos de velocidad, reflexión izquierda/derecha, escala y rotación;
- BiLSTM bidireccional compacta de 128.417 parámetros con atención;
- validación cruzada anidada leave-one-signer-out (LOSO);
- entrenamiento e inferencia del clasificador con NVIDIA CUDA y AMP;
- cámara en vivo en modo general o con una fruta objetivo.

El flujo opcional de transferencia conserva además 75 coordenadas MediaPipe crudas
de pose/manos, reutiliza solamente el encoder de movimiento ASL y descarta su cabeza
de vocabulario. No presupone que ASL y LSEC tengan las mismas palabras.

MediaPipe/OpenCV extraen landmarks en CPU porque el paquete Python utilizado no
expone ese cálculo mediante CUDA. PyTorch sí ejecuta obligatoriamente en GPU tanto
el entrenamiento como el ensemble de inferencia del flujo recomendado.

## Convención del dataset

Cada video debe seguir este patrón:

```text
<nombre_de_fruta><signante>[_tag].mp4
```

El último dígito antes de los tags identifica al signante (`0` a `4`). Ejemplos:

```text
NARANJILLA/Naranjilla0.mp4  -> signer_0, incluido
GUABA/Guaba1_r.mp4          -> signer_1, variante incluida
CIRUELA/Ciruela0_x.MP4      -> signer_0, excluido por defecto
```

El tag `_x` marca una toma manualmente revisada como muy diferente. No se elimina:
queda registrada en `manifest.json`, pero no participa en preparación ni
entrenamiento salvo que se pase `--include-excluded`. Esto permite comparar la
política sin perder trazabilidad.

Comprueba el inventario:

```bash
pfrutas inspect --raw-dir raw
```

## Por qué los folds son por signante

Dividir frames o videos al azar haría que la misma persona aparezca en train y test,
inflando artificialmente la exactitud. PFrutas usa cinco folds externos LOSO: en cada
fold se reserva un signante completo. Dentro del train externo se ejecutan otros
cuatro folds LOSO para escoger la época, sin consultar al test externo.

Este protocolo es más exigente, pero responde a la pregunta útil: “¿funciona con una
persona que el modelo no vio?”. Con solo cinco signantes, las métricas siguen teniendo
incertidumbre y no se puede garantizar 80 % hasta medirlo con personas nuevas.

## Pipeline de features v2

Cada frame conserva 527 valores estáticos:

- 510 coordenadas geométricas normalizadas;
- 13 visibilidades de pose;
- 4 máscaras binarias: pose, cara, mano izquierda y mano derecha.

Se agregan 510 deltas geométricos válidos, para una entrada de 1.037 valores por paso.
Los deltas solo se calculan si el grupo existe en dos frames consecutivos. Los gaps
internos de hasta tres frames se interpolan; gaps largos, extremos ausentes y sus
deltas permanecen exactamente en cero.

Antes de clasificar se detecta el intervalo activo usando energía de brazos, muñecas
y dedos. Luego se remuestrea a 48 posiciones de fase. Esto reduce la sensibilidad a
que una persona haga la seña más rápida, lenta o con una pausa. Durante entrenamiento
se aplican deformaciones temporales, reflexión coherente que intercambia manos,
rotación de ±7° y escala 0,9–1,1. En inferencia se promedian vistas temporales
deterministas.

La geometría global sigue normalizada por hombros y la geometría local de cada mano
por la palma. Así se toleran diferencias moderadas de amplitud sin borrar por completo
la ubicación o extensión del movimiento, que podrían ser discriminativas.

## Mejora 4: transferencia desde ASL Citizen

La primera fuente elegida es el ST-GCN oficial de
[ASL Citizen](https://github.com/microsoft/ASL-citizen-code). Es preferible a un
modelo genérico de acciones para esta etapa porque fue preentrenado con señas y usa
el mismo tipo de entrada MediaPipe: `x/y`, 128 frames y un grafo de 27 nodos de
cuerpo/manos. Se transfieren configuraciones y dinámica, no etiquetas ASL.

La arquitectura nueva combina:

- la representación de 160 dimensiones de la BiLSTM compacta local;
- el embedding de 256 dimensiones del ST-GCN, sin su clasificador de 2.731 señas;
- dos proyecciones a 96 dimensiones y una compuerta aprendida, inicializada con 75 %
  de confianza hacia la rama local ya validada;
- el encoder ASL congelado durante cinco épocas y solo su último bloque abierto desde
  la época 6 con LR `1e-5`; sus BatchNorm permanecen congeladas.

`processed_v2` no contiene los puntos exactos pose 2/5 ni las coordenadas originales,
por lo que no se inventan. El comando transfer vuelve a procesar `raw` una sola vez y
crea `data/processed_asl_transfer_v1`: cada `.npz` contiene las features v2 y
`pose_hands_xy [T,75,2]`. El baseline y sus artefactos no se modifican.

Los pesos no se versionan. El descargador exige aceptar explícitamente los
[términos de investigación de Microsoft](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/),
valida SHA-256 del ZIP y del `.pt`, usa carga segura `weights_only=True`, importa
estrictamente `encoder.*` y descarta `decoder.*`.

## Instalación

Se admite Python 3.10–3.12. Con `uv`:

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate
```

PyTorch debe tener soporte CUDA compatible con el controlador NVIDIA. Verifica una
operación real en GPU:

```bash
python scripts/00_check_cuda.py
```

No hay fallback silencioso en los scripts GPU: si CUDA no está disponible, fallan al
inicio.

## Ejecución rápida

Prepara los 72 videos aceptados a 30 FPS:

```bash
pfrutas prepare
```

Esto crea `data/processed_v2`. Para regenerar todo:

```bash
pfrutas prepare --overwrite
```

Para incluir deliberadamente los cuatro `_x` en un experimento separado:

```bash
pfrutas prepare \
  --include-excluded \
  --output-dir data/processed_v2_with_x
```

Entrena el modelo recomendado con CUDA:

```bash
pfrutas train
```

Los defaults usan cinco folds, 30 épocas internas, `compact_v2`,
`data/processed_v2` y `artifacts/lstm_compact_v2_gpu`. La forma explícita es:

```bash
pfrutas train \
  --processed-dir data/processed_v2 \
  --run-dir artifacts/lstm_compact_v2_gpu \
  --folds 5 \
  --epochs 30 \
  --sequence-length 48 \
  --samples-per-video 4 \
  --batch-size 32 \
  --model-preset compact_v2 \
  --device cuda
```

Una ejecución interrumpida puede continuar solo si configuración, datos, layout y
arquitectura coinciden exactamente:

```bash
pfrutas train --resume
```

Para ejecutar la mejora 4 en un experimento aislado:

```bash
# 1. Cache exacto pose/manos; MediaPipe/OpenCV en CPU
python scripts/05_prepare_asl_transfer.py

# 2. Peso oficial; requiere confirmar los términos enlazados
python scripts/06_download_asl_citizen_weights.py \
  --accept-research-license

# 3. Mismo nested LOSO, fusión y ST-GCN en CUDA/AMP
python scripts/07_train_asl_transfer_gpu.py
```

La corrida se escribe por defecto en
`artifacts/fusion_asl_citizen_stgcn_v1_gpu`. Cada inner/outer/final empieza desde el
mismo checkpoint ASL; ningún fold hereda pesos ajustados por otro signante.

## Probar en vivo

Modo general: no se indica una fruta y el modelo intenta reconocer cualquiera de las
16 clases.

```bash
pfrutas live
```

Modo objetivo:

```bash
pfrutas live --fruit MANGO
pfrutas live --fruit "TOMATE DE ARBOL"
```

Los nombres toleran diferencias de tildes, espacios y guiones. Para listarlos:

```bash
pfrutas live --list-fruits
```

En modo general la interfaz muestra la fruta predicha, confianza, top-k, precisión de
esa clase y métricas CV generales. En modo `--fruit` oculta toda información de otras
frutas y del modelo general: solo muestra probabilidad/confianza, umbral,
precision/recall/F1 CV, resultado y accuracy de sesión de la fruta solicitada.

Controles: `Espacio` registra el intento y limpia el gesto, `R` reinicia la sesión,
y `Q` o `Esc` cierra la aplicación.

## Despliegue en Jetson Orin NX 16 GB

Existe un perfil separado para JetPack 6, ARM64 y cámaras USB/CSI. Genera un bundle
portable con hashes, comprueba CUDA/MediaPipe/OpenCV antes de iniciar y permite medir
los perfiles `full` (seis miembros) y `final` (un miembro). MediaPipe extrae los
landmarks en CPU; el clasificador se ejecuta con CUDA y FP16.

La instalación, construcción del bundle, preflight, benchmark y comandos de cámara
están en [deploy/orin_nx16/README.md](../deploy/orin_nx16/README.md). La plataforma
reproducible principal es JetPack 6.1/Ubuntu 22.04/Python 3.10; JetPack 6.2.x se
admite cuando ya dispone de un PyTorch CUDA funcional.

Para una Orin NX 16 GB que deba usar Ubuntu 20.04, se añadió un perfil de
compatibilidad **experimental** para JetPack 5.1.5/L4T 35.6.1 y Python 3.8:
`deploy/orin_nx16/install_runtime_jp515_ubuntu2004.sh`. No funciona sobre Ubuntu
20.04 genérico; requiere el BSP NVIDIA correspondiente y se valida con
`scripts/11_check_orin_nx16.py --runtime-profile jetpack5`. La guía técnica deja
explícita la comprobación de CUDA, MediaPipe 0.10.5, hashes y paridad antes de usar
la cámara.

Para una guía directa de uso diario —PC NVIDIA, creación/copia del bundle y Jetson
USB/CSI— consulta [GUIA_USO_PC_Y_JETSON.md](../deploy/orin_nx16/README.md).

## Reproducir manualmente el experimento

Estos son los archivos Python autoritativos, en orden:

```bash
# 0. Verificar CUDA, cuDNN, GPU y una operación real
python scripts/00_check_cuda.py

# 1. Extraer landmarks v2; CPU para MediaPipe/OpenCV
python scripts/01_prepare_dataset.py \
  --raw-dir raw \
  --output-dir data/processed_v2 \
  --fps 30

# 2. Nested LOSO + refits; PyTorch CUDA y AMP
python scripts/02_train_nested_cv_gpu.py \
  --processed-dir data/processed_v2 \
  --run-dir artifacts/lstm_compact_v2_gpu \
  --folds 5 \
  --epochs 30 \
  --model-preset compact_v2

# 3. Verificar huellas y recalcular métricas OOF
python scripts/03_evaluate_oof.py \
  --processed-dir data/processed_v2 \
  --run-dir artifacts/lstm_compact_v2_gpu

# 4. Cargar cámara y ensemble en CUDA
python scripts/04_live_inference_gpu.py \
  --run-dir artifacts/lstm_compact_v2_gpu

# 5. Cache multimodal exacto para la mejora 4
python scripts/05_prepare_asl_transfer.py

# 6. Descargar/verificar el preentrenamiento ASL Citizen
python scripts/06_download_asl_citizen_weights.py --accept-research-license

# 7. Entrenar la fusión transfer con nested LOSO y CUDA
python scripts/07_train_asl_transfer_gpu.py

# 8. Verificar huellas y métricas OOF de la corrida transfer
python scripts/03_evaluate_oof.py \
  --processed-dir data/processed_asl_transfer_v1 \
  --run-dir artifacts/fusion_asl_citizen_stgcn_v1_gpu

# 9. Probar el ensemble transfer en vivo, también en CUDA
python scripts/04_live_inference_gpu.py \
  --run-dir artifacts/fusion_asl_citizen_stgcn_v1_gpu
```

Para regenerar las curvas ROC multicategoría desde las probabilidades OOF ya
verificadas, sin reentrenar:

```bash
python scripts/03_evaluate_oof.py \
  --processed-dir data/processed_asl_transfer_v1 \
  --run-dir artifacts/fusion_asl_citizen_stgcn_v1_gpu \
  --write-roc-artifacts
```

Evaluación de una fruta concreta:

```bash
python scripts/03_evaluate_oof.py \
  --processed-dir data/processed_asl_transfer_v1 \
  --run-dir artifacts/fusion_asl_citizen_stgcn_v1_gpu \
  --fruit MANGO

python scripts/04_live_inference_gpu.py \
  --run-dir artifacts/fusion_asl_citizen_stgcn_v1_gpu \
  --fruit MANGO
```

Mapa de archivos:

| Archivo | Responsabilidad |
| --- | --- |
| `scripts/00_check_cuda.py` | Comprueba el entorno NVIDIA/CUDA. |
| `scripts/01_prepare_dataset.py` | Audita nombres, excluye `_x` y genera landmarks v2. |
| `scripts/02_train_nested_cv_gpu.py` | Ejecuta nested LOSO y el modelo final en GPU. |
| `scripts/03_evaluate_oof.py` | Recalcula métricas generales o por fruta y puede regenerar las ROC OOF sin entrenar. |
| `scripts/04_live_inference_gpu.py` | Ejecuta el ensemble con cámara y CUDA. |
| `scripts/05_prepare_asl_transfer.py` | Genera features v2 + pose/manos xy sin tocar `processed_v2`. |
| `scripts/06_download_asl_citizen_weights.py` | Descarga, valida hashes y extrae el checkpoint de forma segura. |
| `scripts/07_train_asl_transfer_gpu.py` | Ejecuta nested LOSO con la fusión BiLSTM + ST-GCN. |
| `src/pfrutas/data.py` | Parser, manifiesto y extracción de videos. |
| `src/pfrutas/features.py` | Normalización, máscaras, segmentación, vistas y deltas. |
| `src/pfrutas/model.py` | Presets y BiLSTM con atención. |
| `src/pfrutas/asl_transfer.py` | Contrato, preprocesado exacto, ST-GCN y loader seguro. |
| `src/pfrutas/training.py` | Folds, AMP, selección interna, métricas y artefactos. |
| `src/pfrutas/live.py` | Inferencia temporal y visualización en vivo. |

## Artefactos y métricas

La corrida v2 escribe, entre otros:

- `training_config.json`: parámetros, GPU, modelo y huellas;
- `splits.json`: asignación auditable de signantes en cada fold;
- `fold_N/selection.json`: selección interna de época;
- `fold_N/best.pt`: refit del fold externo;
- `oof_predictions.npz`: una predicción externa por video;
- `cv_metrics.json` y `per_class_metrics.csv`: accuracy/precision/recall/F1;
- `confusion_matrix.csv/png`: errores por clase;
- `roc_curves.png`, `roc_auc_per_class.csv` y `roc_metrics.json`: curvas y AUC one-vs-rest calculadas únicamente sobre OOF;
- `final_model.pt`: refit con todos los datos para despliegue;
- `ensemble.json`: contrato consumido por el modo en vivo.

Las métricas reportables son las OOF de `cv_metrics.json`. La accuracy de
entrenamiento del modelo final no estima generalización. Los artefactos v1 permanecen
como referencia histórica local y se rechazan explícitamente desde el pipeline v2.

### Resultado reproducido de la corrida v2

La corrida `artifacts/lstm_compact_v2_gpu` se ejecutó realmente en una NVIDIA GeForce
RTX 4070 Laptop con CUDA 13.0, cuDNN 92000 y AMP FP16. Sobre una única predicción OOF
por cada uno de los 72 videos aceptados obtuvo:

- accuracy: **56,94 %**;
- precisión macro: **59,56 %**;
- recall macro: **56,04 %**;
- F1 macro: **55,91 %**;
- accuracy top-3: **72,22 %**.

Las épocas seleccionadas internamente fueron 14, 14, 21, 29 y 24; el refit final usó
la mediana, 21. Las accuracies externas por signante fueron 60,00 %, 62,50 %, 37,50 %,
78,57 % y 50,00 %. Esta dispersión importa más que el mejor fold y confirma que hacen
falta más personas para estimar con precisión la generalización.

Por clase, por ejemplo, MANGO obtuvo 75,00 % precision y 60,00 % recall;
NARANJILLA obtuvo F1 83,33 %. ORITO no tuvo aciertos OOF y UVILLA/TORONJA quedaron
por debajo de 30 % F1, por lo que esas son prioridades claras para la próxima etapa de
recolección. Los soportes son de solo 3–5 videos por fruta, así que estas cifras tienen
alta incertidumbre.

La referencia histórica v1 alcanzó 51,95 % accuracy y 49,11 % F1 macro. La v2 mejora
aproximadamente 5 y 6,8 puntos respectivamente, pero la comparación es exploratoria:
el inventario y la política de exclusión también cambiaron, por lo que no debe
presentarse como una ablación causal limpia.

### Resultado reproducido de transferencia ASL Citizen

La corrida completa `artifacts/fusion_asl_citizen_stgcn_v1_gpu` se entrenó con CUDA
AMP FP16 en una Tesla V100 de 16 GiB. Aplicó nested LOSO sobre los cinco signantes,
seleccionó internamente las épocas y realizó el refit final durante 27 épocas. Sobre
las 72 predicciones OOF externas obtuvo:

- accuracy: **81,94 %**;
- precisión macro: **84,67 %**;
- recall macro: **81,35 %**;
- F1 macro: **81,15 %**;
- accuracy top-3: **90,28 %**.

Por ejemplo, MANGO obtuvo 80 % de precision, recall y F1 (soporte 5), mientras que
NARANJILLA logró recall 100 % y F1 83,33 %. GUABA y TORONJA siguen siendo las clases
más débiles (F1 50 % y 57,14 %). Es una estimación LOSO mucho más prometedora que la
referencia v2, pero los soportes por clase siguen siendo de 3 a 5 videos; hace falta
recoger más personas y condiciones antes de tratar esta cifra como una garantía de
desempeño en producción.

## Pruebas

```bash
python -m pytest -q
```

Las pruebas cubren nombres/tags, exclusión `_x`, separación por signante, layout,
máscaras, interpolación, deltas, reflexión, segmentación, velocidad y presets legados.

## Límites y camino hacia 80 %

El flujo v2 ataca tres fuentes claras de error —sobreajuste, velocidad y lateralidad—,
pero 72 muestras para 16 clases siguen siendo muy pocas. Para validar o superar 80 %
de manera creíble, la mejora de mayor impacto pendiente es recolectar más videos por
fruta, signante y condición, además de una clase `NO_GESTURE`. Cualquier ajuste futuro
de arquitectura o aumentos debe seleccionarse dentro de los folds internos; mirar el
test externo para elegir variantes produciría una cifra optimista.
