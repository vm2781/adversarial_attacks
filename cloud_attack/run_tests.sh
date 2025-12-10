#!/bin/bash

SOURCE_MODELS=("lenet" "squeezenet" "lenet_pgd" "squeezenet_pgd" "lenet_mifgsm" "squeezenet_mifgsm" "lenet_pixle")
ATTACK_TYPES=("pgd" "mifgsm" "pixle")
NUM_EXAMPLES=100

for model in "${SOURCE_MODELS[@]}"; do
    for attack in "${ATTACK_TYPES[@]}"; do
        echo "=== Model: $model, Attack: $attack ==="
        curl -s -X POST https://adversarial-mnist-717230938900.us-central1.run.app/generate \
            -H "Content-Type: application/json" \
            -d "{\"source_model\": \"$model\", \"attack_type\": \"$attack\", \"num_examples\": $NUM_EXAMPLES}"
        echo -e "\n"
    done
done