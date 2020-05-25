Handcrafted [UNet](https://arxiv.org/abs/1505.04597) `pytorch` training script. Uses the [Segmented Bob Ross Images](https://www.kaggle.com/residentmario/segmented-bob-ross-images) dataset on Kaggle.

To run on Spell (requires being in the spellrun org for the GH integration):

```bash
prodspell run \
  --machine-type V100 \
  --github-url https://github.com/ResidentMario/spell-unet-bob-ross.git \
  --tensorboard-dir /spell/tensorboards/model_1/ \
  --pip kaggle --pip Pillow \
  "chmod +x /spell/scripts/download_data.sh; /spell/scripts/download_data.sh; python /spell/models/model_1.py"
```
