# Deep Multi-View Subspace Clustering with Anchor Graph
Deep Multi-View Subspace Clustering with Anchor Graph is a paper accepted by IJCAI2023. 
 

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Description
We propose a novel deep self-supervised model for
MVSC. A unified target distribution is generated via spectral clustering which is more robust and can accurately guide the feature learning process. The target distribution and learned features are updated iteratively. To boost the model efficiency, we use anchor graph to construct the graph matrix. This strategy can significantly reduce time complexity by sampling anchor points. Besides, we utilize contrastive learning on pseudo-labels to alleviate the conflict between the consistency objective and the reconstruction objective, thus consistent soft cluster assignments can be obtained among multiple views.


## Installation

 

```bash
pip install -r requirements.txt
```

## Usage

To train the fully connected layer  model using the specified arguments, you can run the following command:

```bash
python train.py --lr 0.001 --n_clusters 7 --n_z 10 --dataset BDGP --arch 50 --gamma 5 --noise 0

 
 Training a model with convolutional layers is the same as training a model with fully connected layers.
 

 

 

## Citation
@inproceedings{cui2023deep,
  title={Deep Multi-View Subspace Clustering with Anchor Graph},
  author={Cui, Chenhang and Ren, Yazhou and Pu, Jingyu and Pu, Xiaorong and He, Lifang},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2023}
}

 
