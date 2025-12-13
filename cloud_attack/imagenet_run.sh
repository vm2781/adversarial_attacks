#!/bin/bash
SEED=$1
ATTACK_TYPES=("pgd" "mifgsm" "pixle")
NUM_EXAMPLES=25

for attack in "${ATTACK_TYPES[@]}"; do
    echo "=== ImageNette Attack: $attack ==="
    curl -s -X POST https://adversarial-mnist-717230938900.us-central1.run.app/imagenette \
        -H "Content-Type: application/json" \
        -d "{\"attack_type\": \"$attack\", \"seed\": $SEED, \"num_examples\": $NUM_EXAMPLES}"
    echo -e "\n"
done