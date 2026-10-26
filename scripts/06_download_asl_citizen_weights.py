"""Descarga y verifica el encoder preentrenado oficial de ASL Citizen.

El código de ASL Citizen es MIT, pero Microsoft publica los materiales del
dataset/modelo bajo términos de investigación. Este script exige una aceptación
explícita y nunca añade los pesos al repositorio.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


CHECKPOINT_URL = (
    "https://github.com/microsoft/ASL-citizen-code/releases/download/"
    "checkpoints_v1/ASL_citizen_stgcn_weights.zip"
)
LICENSE_URL = "https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/"
ZIP_SHA256 = "4b52578ed24ed975756c6447c56ba9bb90576eb302518c9b3aef955bfb20eb01"
CHECKPOINT_SHA256 = "b08d84afb5a0fdf4c723fd7be2db897b2490a921ccf538cff1b345f905934d68"
ARCHIVE_MEMBER = "ASL_citizen_stgcn_weights.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    candidates: list[zipfile.ZipInfo] = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Ruta insegura dentro del ZIP: {member.filename!r}")
        if not member.is_dir() and path.name == ARCHIVE_MEMBER:
            candidates.append(member)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Se esperaba exactamente {ARCHIVE_MEMBER!r} dentro del ZIP."
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/ASL_citizen_stgcn_weights.pt"),
    )
    parser.add_argument(
        "--accept-research-license",
        action="store_true",
        help="Confirma que leíste y aceptas los términos enlazados de Microsoft.",
    )
    args = parser.parse_args()

    if not args.accept_research_license:
        parser.error(
            "Debes leer los términos y pasar --accept-research-license: "
            f"{LICENSE_URL}"
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and _sha256(output) == CHECKPOINT_SHA256:
        print(f"Checkpoint ya verificado: {output}")
        return

    archive_path = output.with_suffix(".zip.part")
    checkpoint_part = output.with_suffix(output.suffix + ".part")
    try:
        print(f"Descargando checkpoint oficial desde {CHECKPOINT_URL}")
        with urllib.request.urlopen(CHECKPOINT_URL) as response, archive_path.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        actual_zip_hash = _sha256(archive_path)
        if actual_zip_hash != ZIP_SHA256:
            raise RuntimeError(
                "SHA256 del ZIP inesperado: "
                f"{actual_zip_hash}; esperado {ZIP_SHA256}."
            )
        with zipfile.ZipFile(archive_path) as archive:
            member = _safe_member(archive)
            with archive.open(member) as source, checkpoint_part.open("wb") as target:
                shutil.copyfileobj(source, target)
        actual_checkpoint_hash = _sha256(checkpoint_part)
        if actual_checkpoint_hash != CHECKPOINT_SHA256:
            raise RuntimeError(
                "SHA256 del checkpoint inesperado: "
                f"{actual_checkpoint_hash}; esperado {CHECKPOINT_SHA256}."
            )
        checkpoint_part.replace(output)
    finally:
        archive_path.unlink(missing_ok=True)
        checkpoint_part.unlink(missing_ok=True)

    print(f"Checkpoint verificado: {output}")
    print(f"SHA256: {CHECKPOINT_SHA256}")


if __name__ == "__main__":
    main()
