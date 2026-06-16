# Dataset Timing Reference

Measured on Jetson Orin, ViT-B/16 backbone, `batch_size=1`, ~46 it/s.
Use these to estimate how long to wait after launching a job via `--datasets`.

## CD Benchmark Datasets

| `--datasets` name  | Samples | Approx. time |
|--------------------|---------|--------------|
| caltech101         | 2465    | ~53s         |
| oxford_flowers     | 2463    | ~54s         |
| oxford_pets        | 3669    | ~79s         |
| dtd                | —       | not yet timed |
| eurosat            | —       | not yet timed |
| fgvc               | —       | not yet timed |
| food101            | —       | not yet timed |
| stanford_cars      | —       | not yet timed |
| sun397             | —       | not yet timed |
| ucf101             | —       | not yet timed |

## OOD Benchmark Datasets

| `--datasets` name  | Samples | Approx. time |
|--------------------|---------|--------------|
| I (ImageNet)       | —       | not yet timed |
| A (ImageNet-A)     | —       | not yet timed |
| V (ImageNetV2)     | —       | not yet timed |
| R (ImageNet-R)     | —       | not yet timed |
| S (ImageNet-Sketch)| —       | not yet timed (exceeded 3h wall-time in prior runs) |

## Notes

- Throughput is consistent at ~46 it/s across datasets on this hardware.
- For untimed datasets, use `n_samples / 46` seconds as a rough estimate.
- For multi-dataset runs, sum individual times and add ~30s startup overhead.
- The default `run_cd_benchmark_vit.sh` (caltech101 + oxford_flowers + oxford_pets) takes roughly **3 minutes total**.
- ResNet-50 timing may differ; update this file when timed.
