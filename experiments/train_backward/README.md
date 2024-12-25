# Code for training backward LM

## Preprocess

```sh
bash preprocess_sample.sh
```

It will generate data files such as `tokenized/wikitext-103-raw-v1/train/data-00000-of-00004.arrow` under `/temp`.
Make sure you copy it to your working directory.


# Training

```sh
bash train.sh
```

```sh
accelerate launch train.py \
    --train_dataset_dir data/tokenized/$dataset_id/train/ \
    --valid_dataset_dir data/tokenized/$dataset_id/validation/ \
    --arch_id gpt2 \
    --tokenizer_id meta-llama/Llama-2-7b-chat-hf \
    --outdir models/wiki+book/seed$seed \
    --epochs 40 \
    --seed 11
```

- `--train_dataset_dir`,  `--valid_dataset_dir`: set directories each of which contain `data-*.arrow` files from preprocessing
- `arch_id`: Load a model architecture/size configuration from predefined IDs.
- `tokenizer_id`: Load vocabulary from a model ID.

The above example show the case you use GPT-2 architecture but emplay Llama2 vocabulary.


# Converting an old pickle model to a safetensors file

The above code (with the older `transformers` library) result in an old pickle model with `pytorch_model.bin`.
`pytorch_model.bin` is slow to load generally considered not safe (can run arbitary code), hence we converted models to safetensors before making them public.
You can run the following command to convert `pytorch_model.bin` to `model.safetensorse:

```bash
python convert_to_safetensors.py path/to/your/model
```

This will generate `path/to/your/model/model.safetensors`.
You can manually delete `path/to/your/model/pytorch_model.bin`.
