# Deep Multi-View Subspace Clustering with Anchor Graph
Deep Multi-View Subspace Clustering with Anchor Graph (DMCAG) is a paper accepted by IJCAI2023. 
 

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#license)

## Project Description
We propose a novel deep self-supervised model for MVSC. A unified target distribution is generated via spectral clustering which is more robust and can accurately guide the feature learning process. The target distribution and learned features are updated iteratively. To boost the model efficiency, we use anchor graph to construct the graph matrix. This strategy can significantly reduce time complexity by sampling anchor points. Besides, we utilize contrastive learning on pseudo-labels to alleviate the conflict between the consistency objective and the reconstruction objective, thus consistent soft cluster assignments can be obtained among multiple views.


## Installation

 

```bash
pip install -r requirements.txt
```

## Usage



# Prepare your dataset
The dataset would be a single .mat format file in a structure as the following:
```
dataset.mat
├── Y    1 * n
├── X1   n * d_1
├── X2   n * d_2
...   
```
Then place the file inside a folder named "data" and set up your dataset as follows in training file：
```
    if args.dataset ==  dataset_name:
        args.n_input = [d_1, ...]
        args.viewNumber =  v
        args.instanceNumber = n
        args.batch_size =  x
        args.n_clusters =  k
        args.arch= p
        args.gamma= q
        args.save_path = path/to/save/parameters
```


To train the fully connected layer  model using the specified arguments, you can run the following command:

```bash
python train_fc.py --lr 0.001 --n_z 10 --dataset BDGP --arch 50 --gamma 5  --dataset  dataset_name
```

 You will get the clustering results as follows:
 
 ```bash
 Acc 0.8564, nmi 0.7321, ari 0.6459
 ```
 Training a model with convolutional layers is the same as training a model with fully connected layers.
 

 

 

## Citation
If you're using DMCAG in your research or applications, please cite using this BibTeX:
```bibtex
@inproceedings{cui2023deep,
  title={Deep Multi-View Subspace Clustering with Anchor Graph},
  author={Cui, Chenhang and Ren, Yazhou and Pu, Jingyu and Pu, Xiaorong and He, Lifang},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2023}
}
```
 
