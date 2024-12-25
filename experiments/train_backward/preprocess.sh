TOKENIZER_NAME=meta-llama/Llama-2-7b-hf

source ../.venv/bin/activate

python3 make_tokenized_data.py \
    --tokenizer_name=$TOKENIZER_NAME \
    --dataset_name wikitext \
                   bookcorpus \
    --dataset_config_name wikitext-103-raw-v1 \
                          None \
    --preprocessing_num_workers 20 \
    --base_save_dir /tmp/goto/data \
    --raw_dir raw/wikipedia_20220301_en \
              raw/bookcorpus \
    --tokenized_dir tokenized/wikipedia_20220301_en \
                    tokenized/bookcorpus \
    --output_dir dummy \
    --log_level debug

deactivate
