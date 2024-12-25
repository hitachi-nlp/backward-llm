Concatenated language model


# How to train a model
```sh
accelerate launch train.py \
    --name_or_path_forward gpt2 \
    --name_or_path_backward path/to/backward_model \
    --dataset_name conll2003 \
    --task_name ner \
    --representation concat \
    --outdir outputs/sample \
    --lr 1e-3 \
    --dropout 0.1 \
    --batch_size 32 \
    --epochs 10 \
    --seed 10 \
    --few -1 \
    --lr_scheduler_type linear
```

- `task_name` can be `ner`, `pos`, `chunk` when `--dataset_name conll2003`.
- `representation` can be `concat`, `forward`.
    - `concat` uses concatenated representation of the forward and backward models.
    - `forward` uses only the representation of the forward model.

The output directory `outputs/sample` contains `best/` and `last/` directory, and `log.json`.
- The best/ is the checkpoint that has minimum loss on the validation.
    - This directory also contains `output.txt` for the visualization of the prediction.
- The last/ is the last checkpoint in the training.
- `log.json` contains the loss, accuracy, F1 scores on the validation set and test set.

### Few-shot setting
You can use `--few` option to specify K for K-shot learning (default: -1).  
- -1 means full-shot setting.
- Otherwise, K-shot learning will be performed. 

### Hyperparameter sampling
The procedure of the experiments with the sampled hyperparameters (e.g. learning rate).

1. Prepare parameter candidates
This is JSON format: like `Dict[str, list]` which means `{key: candidates for the key}`. For example,
```python
{
    "batch_size": [4],
    "lr": [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4],
    "seed": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "dropout": [0, 0.1, 0.2, 0.3]
}
```

2. Make the shell for qsub with `utils/draw_params.py`.  
```shell
LR=`python utils/draw_param.py --json <Path to the JSON> --key lr`
SEED=`python utils/draw_param.py --json <Path to the JSON> --key seed`

python example.py \
    --lr $LR \
    --seed $SEED
    ...
```