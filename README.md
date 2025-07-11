
`pip install -r requirements.txt`

The current dataset format is `JSON`, not `JSONL`, which means you need to convert the source file to a JSON file. The script for the conversion is `convert_jsonl.py`.


All set, simply run `sh scripts/finetune_llama2_13b.sh`
> Before that, you may want to configure wandb and huggingface correctly, e.g., `huggingface-cli login --token`
