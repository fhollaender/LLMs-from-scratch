"""SLURM-/GPU-Hilfe fuer LRZ-Cluster und PyTorch-Notebooks."""

from __future__ import annotations

import os
import subprocess
from typing import Literal

import torch

GpuType = Literal["v100", "a100", "h100", "p100", "test"]

# Haeufig genutzte LRZ-GPU-Partitionen (Fallback, falls sinfo nicht erreichbar ist).
LRZ_GPU_PARTITIONS: dict[GpuType, str] = {
    "v100": "lrz-v100x2",
    "a100": "lrz-hgx-a100-80x4",
    "h100": "lrz-hgx-h100-94x4",
    "p100": "lrz-hpe-p100x4",
    "test": "test-v100x2",
}


def _run(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def list_gpu_partitions() -> list[dict[str, str]]:
    """Gibt verfuegbare LRZ-GPU-Partitionen mit Groesse und Status zurueck."""
    try:
        output = _run(["sinfo", "-h", "-o", "%P|%G|%D|%T"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [
            {
                "partition": partition,
                "gpu_type": gpu_type,
                "nodes": "?",
                "state": "unknown",
            }
            for gpu_type, partition in LRZ_GPU_PARTITIONS.items()
        ]

    partitions: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in output.splitlines():
        name, gres, nodes, state = line.split("|")
        name = name.rstrip("*")
        if not name.startswith("lrz-") or not gres.startswith("gpu"):
            continue
        if name in seen:
            continue
        seen.add(name)
        partitions.append(
            {
                "partition": name,
                "gres": gres,
                "nodes": nodes,
                "state": state,
            }
        )
    return partitions


def _is_login_node() -> bool:
    hostname = os.uname().nodename
    return hostname.startswith("login") or hostname.startswith("cm")


def _in_slurm_job() -> bool:
    return bool(os.environ.get("SLURM_JOB_ID"))


def _print_slurm_context() -> None:
    if not _in_slurm_job():
        print("Kein aktiver SLURM-Job erkannt.")
        return

    keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NODELIST",
        "SLURM_GPUS_ON_NODE",
        "SLURM_GPUS",
        "CUDA_VISIBLE_DEVICES",
    )
    print("SLURM-Job aktiv:")
    for key in keys:
        value = os.environ.get(key)
        if value:
            print(f"  {key}={value}")


def build_srun_command(
    partition: str,
    gpus: int = 1,
    cpus_per_task: int = 4,
    mem: str = "32G",
    time: str = "08:00:00",
) -> str:
    """Erzeugt den srun-Befehl fuer eine interaktive GPU-Session am LRZ."""
    return (
        f"srun --partition={partition} --gres=gpu:{gpus} "
        f"--cpus-per-task={cpus_per_task} --mem={mem} --time={time} "
        "--pty bash -l"
    )


def setup_training_device(
    gpu_type: GpuType = "v100",
    partition: str | None = None,
    gpu_index: int = 0,
    require_gpu: bool = True,
    cpus_per_task: int = 4,
    mem: str = "32G",
    time: str = "08:00:00",
) -> torch.device:
    """
    Notebook-Setup fuer GPU-Training auf dem LRZ-Cluster.

    Am Anfang eines Notebooks ausfuehren:

        from utils.slurm_gpu import setup_training_device
        device = setup_training_device(gpu_type="v100")

    Wenn der Kernel bereits in einer SLURM-GPU-Session laeuft, wird die GPU
    erkannt und ein passendes torch.device zurueckgegeben. Auf einem Login-Node
    ohne GPU wird der passende srun-Befehl ausgegeben.
    """
    selected_partition = partition or LRZ_GPU_PARTITIONS[gpu_type]

    if torch.cuda.is_available():
        _print_slurm_context()
        device = torch.device(f"cuda:{gpu_index}")
        gpu_name = torch.cuda.get_device_name(gpu_index)
        gpu_count = torch.cuda.device_count()
        print(f"CUDA verfuegbar: {gpu_count} GPU(s), nutze {device} ({gpu_name}).")
        return device

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Keine CUDA-GPU, nutze Apple MPS: {device}.")
        return device

    print("Verfuegbare LRZ-GPU-Partitionen:")
    for entry in list_gpu_partitions()[:8]:
        print(
            f"  - {entry['partition']}: {entry.get('gres', '?')} "
            f"({entry.get('nodes', '?')} nodes, {entry.get('state', '?')})"
        )

    srun_cmd = build_srun_command(
        partition=selected_partition,
        cpus_per_task=cpus_per_task,
        mem=mem,
        time=time,
    )

    location = "Login-Node" if _is_login_node() else "Knoten ohne GPU"
    print(f"\n{location}: PyTorch sieht aktuell keine GPU.")
    print("Starte zuerst eine interaktive SLURM-Session im Terminal:")
    print(f"\n  {srun_cmd}\n")
    print("Danach Jupyter in dieser Session starten und dieses Notebook neu oeffnen.")
    print(f"Empfohlene Partition fuer gpu_type='{gpu_type}': {selected_partition}")

    if require_gpu:
        raise RuntimeError(
            "Keine GPU verfuegbar. Bitte zuerst eine SLURM-GPU-Session starten "
            f"(Partition: {selected_partition})."
        )

    device = torch.device("cpu")
    print(f"Fallback auf CPU: {device}")
    return device
