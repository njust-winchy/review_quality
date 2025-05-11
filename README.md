# review_quality


## Overview

**Dataset and source code for paper "".**



## Model overview

This study proposes a framework consisting of knowledge-guided fusion module and two paralleled sub-tasks. The two paralleled sub-tasks are originality prediction (OP) and decision prediction (DP) respectively.<br>


## Directory structure

<pre>
originality_predict                               Root directory
├── Code                                          Source code folder
│   ├── baseline_model                            Baseline model folder
│   │    ├── load_method.py                       Load data for Review and Method as input
│   │    ├── main.py                              Train the model for Review and Feedback as input
│   │    ├── method_main.py                       Train the model for Review and Method as input
│   │    ├── model.py                             Model structure
│   │    ├── predict.py                           Predict the result
│   │    ├── split_data.py                        Load data for Review and Feedback as input
│   │    ├── util.py                              Data process tool
│   ├── Bi_interaction.py                         Proposed model structure      
│   ├── p_predict.py                              Predict the result
│   └── proposed_main.py                          Train the proposed model 
│   └── read_data.py                              Load data for Review and Feedback
├── Dataset                                       Dataset folder
│   └── Dataset.json                              Preprocessed Dataset
│
└── README.md
</pre>


## Dependency packages
System environment is set up according to the following configuration:
- transformers==4.16.2
- nltk==3.6.7
- matplotlib==3.5.1
- scikit-learn==1.1.3
- pytorch 2.0.1
- tqdm 4.65.0
- numpy 1.24.1

## Acknowledgement

We express our gratitude to the team at openreview.net for their dedication to advancing transparency and openness in scientific communication. We utilized the aspect identifying tool developed by Yuan et al.（2022）(https://github.com/neulab/ReviewAdvisor).

>Yuan, W., Liu, P., & Neubig, G. (2022). Can we automate scientific reviewing?. Journal of Artificial Intelligence Research, 75, 171-212.<br>


## Citation
Please cite the following paper if you use this code and dataset in your work.
