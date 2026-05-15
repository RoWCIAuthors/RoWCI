# RoWCI

## Environment

```bash
pip install -r requirements.txt
```

Model checkpoints:

- Z extraction: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
- WCI embeddings: https://huggingface.co/Qwen/Qwen3-Embedding-8B

Example download commands:

```bash
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir models/Qwen3-30B-A3B-Instruct-2507
huggingface-cli download Qwen/Qwen3-Embedding-8B --local-dir models/Qwen3-Embedding-8B
```

## Steps

1. Prepare source JSONL files.

2. Generate Z coordinates with the local Z extraction model.

3. Convert Z coordinates to RoWCI input format:

```bash
INPUT_JSONL=data/HS/hs1_3000.jsonl \
Z_JSONL=path/to/z_coordinates.jsonl \
SCHEMA_JSON=data/extracted_Z/schema/helpfulness.json \
DATASET=hs \
AXIS=helpfulness \
SOURCE=hs1 \
PROMPT_FILE=path/to/z_extraction_prompt.txt \
OUTPUT_JSONL=data/extracted_Z/hs/helpfulness/extracted_z.jsonl \
bash scripts/run_z_extraction.sh
```

4. Run RoWCI:

```bash
bash scripts/run_rowci.sh
```

If Z files are already prepared, place them at `data/extracted_Z/{dataset}/{axis}/extracted_z.jsonl` and run step 4.
