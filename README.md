# CIFAR10 TorchPie

## Requirements

- Pytorch
- TorchPie
- TensorBoard

Install TorchPie
```bash
pip install git+https://git.dev.tencent.com/SunDoge/torchpie.git
```

## Usage

Use seleted GPUs

```bash
python -m torchpie.distributed.launch --gpus 0,1 main.py -c config/default.conf -e exps/resnet20-2gpu
```

Use all GPUs

```bash
python -m torchpie.distributed.launch main.py -c config/default.conf -e exps/resnet20-2gpu
```

Make sure you use different `experiment-path` every time you run.

## Development

```bash
ln -s ../torchpie/torchpie ./
```