uv run accelerate launch \
    --dynamo_backend=no \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    eval.py \
    --mother ./outputs/fsfdmd \
    --targets $TARGET \
    --checkpoints ${CKPT:-top}
