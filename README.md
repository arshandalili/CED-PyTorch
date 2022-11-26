# CED-Pytorch
An implementation of CED: [Credible Early Detection of Social Media Rumors](https://ieeexplore.ieee.org/document/8939421) (CED) model for Early Fake News Detection in Pytorch.

## Implementation
The implementation is done with Pytorch to make training and use easier on different datasets than the [paper's original implementation](https://github.com/thunlp/CED)
- The implementation is done using the [Pytorch Lightning](https://www.pytorchlightning.ai/) package. 
- Integrated with [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) for hyperparameter tuning.
- Code is cleaner, and more understandable. 

## How to use?

### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Preparing Data

To prepare data, go to the data folder and run prepare_vocab.py to create the vocabulary for the Tokenizer from your dataset. (If your dataset is not in the corresponding format, you should modify the code accordingly.) Then run the prepare.py file to tokenize the dataset.

### Tuning

To fine tune it for the dataset:

```tune
python3 CED_tune.py
```

This will find the best hyperparameters for the dataset.

### Training

To train the model:

```train
python3 CED_train.py
```

This will train the model with the its best hyperparameters on a dataset.

or you can run the CED_train.ipynb:

```train_notebook
python3 CED_train.ipynb
```


### Evaluation

To evaluate the model:

```evaluate
python3 CED_evaluate.ipynb
```

This will give you the result and evaluation of the model on a dataset.


## Citation

If you use CED, please cite the paper as follows:

```
@article{song2019ced,
  title={CED: credible early detection of social media rumors},
  author={Song, Changhe and Yang, Cheng and Chen, Huimin and Tu, Cunchao and Liu, Zhiyuan and Sun, Maosong},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2019},
  publisher={IEEE}
}
```
