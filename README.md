# DualLevelAug

Graph-Guided Dual-Level Augmentation for 3D Scene Segmentation, ACM MM 2025.

## Setup

The following runs S3DIS with PTv3. Requires Linux, an NVIDIA GPU, CUDA 12.4 (including `nvcc`), GCC/G++ 13.2, and OpenBLAS development headers. Run from the repository root.

```sh
conda create -n dla-core python=3.10 -y
conda activate dla-core
python -m pip install -e '.[geometry,graph]'

conda create -n dla-pointcept python=3.10 -y
conda activate dla-pointcept
conda run -n dla-core python scripts/setup_backends.py pointcept --fetch
conda run -n dla-core python scripts/setup_backends.py pointcept --install --python "$CONDA_PREFIX/bin/python"
conda activate dla-core
```

## Run

```sh
cp configs/local.example.json local.json
```

Set `datasets.s3dis.raw_root` to the dataset directory containing `Area_*`, and `backends.pointcept.python` to the Python executable in `dla-pointcept`.

```sh
dla experiments plan table2.ptv3.ours.s3dis --overrides local.json
dla experiments run table2.ptv3.ours.s3dis --overrides local.json
```

`run` prepares data, collects statistics, pretrains the graph encoder, generates scenes, trains PTv3, evaluates, and writes the report. Outputs are saved under `artifacts/`; rerunning reuses completed tasks.
