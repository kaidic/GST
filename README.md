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

For TpuGraphs dataset, download the dataset following instructions [here](https://github.com/google-research-datasets/tpu_graphs), by default, put the `train/valid/test` splits under the folder `/datasets/TPUGraphs/raw/npz/layout/xla/random`. Then run

```bash
python main_tpugraphs.py --cfg configs/tpugraphs.yaml
```


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