To train model:
```bash
python train_model.py --model [resnet18/resnet50]
```

If want robust model:
```bash
python train_model.py --model [resnet18/resnet50] --attack [pgd/cw] --robust
```

Generate Adversarial Examples from MNIST test set:
```bash
python generate_adversarial_images.py --model [resnet18/resnet50] --model_path [PATH_TO_MODEL] --attack [pgd/cw] --output_dir [OUTPUT_DIR]
```