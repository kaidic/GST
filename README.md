## Learning Large Graph Property Prediction via Graph Segment Training
Kaidi Cao, Phitchaya Mangpo Phothilimthana, Sami Abu-El-Haija, Dustin Zelle, Yanqi Zhou, Charith Mendis, Jure Leskovec, Bryan Perozzi
_________________

This is the implementation of GST+EFD in the paper [Learning Large Graph Property Prediction via Graph Segment Training](https://arxiv.org/pdf/2305.12322.pdf) in PyTorch.

### Dependency

The codebase is developed based on [GraphGPS](https://github.com/rampasek/GraphGPS). Installing the environment follwoing its instructions.

### Dataset

- MalNet, the split info of Malnet-Large is provided in splits folder.
- TpuGraphs.
  
### Training 

We provide several training examples with this repo:

```bash
python main.py --cfg configs/malnetlarge-GST.yaml
```

For TpuGraphs dataset, download the dataset following instructions [here](https://github.com/google-research-datasets/tpu_graphs), by default, put the `train/valid/test` splits under the folder `./datasets/TPUGraphs/raw/npz/layout/xla/random`. **To run on other collections, modify `source` and `search` in in [tpu_graphs.py](https://github.com/kaidic/GST/blob/main/graphgps/loader/dataset/tpu_graphs.py)**.

You can train by invoking:

```bash
python main_tpugraphs.py --cfg configs/tpugraphs.yaml
```

Please change `device` from `cuda` to `cpu` in the yaml file if you want to try cpu only training.

To evaluate on TpuGraphs dataset, run

```bash
python test_tpugraphs.py --cfg configs/tpugraphs.yaml
```

If memory is not sufficient, change `batch_size` to 1 during evaluation. Set `cfg.train.ckpt_best` to `True` to save the best validation model during training for further evaluation.

### Reference

If you find our paper and repo useful, please cite as

```
@article{cao2023learning,
  title={Learning Large Graph Property Prediction via Graph Segment Training},
  author={Cao, Kaidi and Phothilimthana, Phitchaya Mangpo and Abu-El-Haija, Sami and Zelle, Dustin and Zhou, Yanqi and Mendis, Charith and Leskovec, Jure and Perozzi, Bryan},
  journal={arXiv preprint arXiv:2305.12322},
  year={2023}
}
```
