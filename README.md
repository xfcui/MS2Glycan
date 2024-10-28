## MS2Glycan: a model for generating glycan structures by mass spectrometry
```
Authors: Defeng Li, Zizheng Nie, Yanmin Liu, Xiaojun Cai, Yong Zhang*,  Xuefeng Cui*
    - *: To whom correspondence should be addressed.
Contact: xfcui@email.sdu.edu.cn
```


### Introduction

We introduce MS2Glycan, a deep learning method for iteratively generating novel glycan structures from mass spectrometry data. It models component generation similarly to an object detection task, incorporating both localization and classification sub-tasks, and constructs the glycan structure component by component. MS2Glycan offers a key improvement over GlycanFinder by using a data-driven approach to predict component locations, rather than relying on predefined rules.

### Usage

MS2Glycan is based on the framework and training principles of GlycanFinder[<sup>1</sup>](#refer-anchor-1), utilizing data from five types of mouse tissues proposed by pGlyco 2.0[<sup>2</sup>](#refer-anchor-2). It employs a five-fold cross-validation approach, where four tissues are used for training and one tissue is used for prediction.  
The five mouse datasets were downloaded from the PRIDE repository with accession numbers: PXD005411 (mouse brain), PXD005412 (mouse kidney), PXD005413 (mouse heart), PXD005553 (mouse liver), PXD005555 (mouse lung). 

#### Run
**This repository provides example data ,model parameters (lung for testing) and one-click code execution.**  

Clone this repository by:
```shell
git clone https://github.com/xfcui/MS2Glycan.git
```

Make sure the python version you use is >= 3.7, and install the packages by:
```shell
bash install.sh
```

Train the model:
**If you are training with your own dataset, you need to provide a glycan library in GlycoCT format and use create_combination function to generate the combination.pkl file.**
```shell
python train.py
```

Evaluate the framework:
```shell
python inference.py
```

Peak matching on the original mass spectrum:
```
match_peaks.ipynb
```



### Reference

<div id="refer-anchor-1"></div>
- [1] Sun W, Zhang Q, Zhang X, et al. Glycopeptide database search and de novo sequencing with PEAKS GlycanFinder enable highly sensitive glycoproteomics[J]. Nature Communications, 2023, 14(1): 4046.  
  
<div id="refer-anchor-2"></div>
- [2] Liu M Q, Zeng W F, Fang P, et al. pGlyco 2.0 enables precision N-glycoproteomics with comprehensive quality control and one-step mass spectrometry for intact glycopeptide identification[J]. Nature communications, 2017, 8(1): 438.