# RCL
# Introduction
Contrastive Learning~(CL) enhances the training of sequential recommendation~(SR) models through informative self-supervision signals. Existing methods often rely on data augmentation strategies to create positive samples and promote representation invariance. Some strategies such as item reordering and item substitution may inadvertently alter user intent. Supervised Contrastive Learning~(SCL) based methods find an alternative for augmentation-based CL methods by selecting same-target sequences~(interaction sequences with the same target item) to form positive samples. However, SCL-based methods suffer from the scarcity of same-target sequences and consequently lack enough signals for contrastive learning.
In this work, we propose to use similar sequences~(with different target items) as additional positive samples and introduce a novel method called \textbf{R}elative \textbf{C}ontrastive \textbf{L}earning (RCL) for sequential recommendation. The proposed RCL comprises a dual-tiered positive sample selection module and a relative contrastive learning module. The former module selects same-target sequences as strong positive samples and selects similar sequences as weak positive samples. The latter module employs a weighted relative contrastive loss, which ensures that each sequence is represented closer to its strong positive samples than its weak positive samples.
We apply RCL on two mainstream deep learning-based SR models, and our empirical results reveal that RCL can achieve 4.88\% improvement averagely than the state-of-the-art SR methods on five public datasets and one private dataset.

# Reference

Please cite our paper if you use this code.

```
@inproceedings{wang2024rcl,
  title={Relative contrastive learning for sequential recommendation
with similarity-based positive pair selection},
  author={Zhikai Wang, Yanyan Shen, Yinghua Zhang, Zexi Zhang, Li He, Hao Gu and Yichun Li},
  booktitle={CIKM},
  year={2024}
}
```

# Implementation
## Requirements

Python >= 3.7  
Pytorch >= 1.2.0  
tqdm == 4.26.0


to train:

```
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```

just inference:

```
python main.py --device=cuda --dataset=ml-1m --train_dir=default --state_dict_path='ml-1m_default/SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true --maxlen=200

```

output for each run would be slightly random, as negative samples are randomly sampled, here's my output for two consecutive runs:

```
1st run - test (NDCG@10: 0.5897, HR@10: 0.8190)
2nd run - test (NDCG@10: 0.5918, HR@10: 0.8225)
```
