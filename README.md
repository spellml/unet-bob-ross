# unet-bob-ross

Handcrafted [UNet](https://arxiv.org/abs/1505.04597) `pytorch` training script. Uses the [Segmented Bob Ross Images](https://www.kaggle.com/residentmario/segmented-bob-ross-images) dataset on Kaggle.

To run code and notebooks in a Spell workspace:

```bash
spell jupyter --lab \
  --github-url https://github.com/ResidentMario/spell-unet-bob-ross.git \
  --pip kaggle \
  --env KAGGLE_USERNAME=YOUR_USERNAME \
  --env KAGGLE_KEY=YOUR_KEY \
  unet-bob-ross
```

To execute the training scripts in a Spell run:

```bash
spell run \
  --machine-type t4 \
  --github-url https://github.com/ResidentMario/spell-unet-bob-ross.git \
  --pip kaggle \
  --env KAGGLE_USERNAME=YOUR_USERNAME \
  --env KAGGLE_KEY=YOUR_KEY \
  --tensorboard-dir /spell/tensorboards/model_3 \
  "chmod +x /spell/scripts/download_data.sh; chmod +x /spell/scripts/upgrade_env.sh; /spell/scripts/download_data.sh; /spell/scripts/upgrade_env.sh; python /spell/models/model_3.py"
```

```bash
spell run \
  --machine-type t4 \
  --github-url https://github.com/ResidentMario/spell-unet-bob-ross.git \
  --pip transformers --pip tokenizers --pip kaggle \
  --env KAGGLE_USERNAME=YOUR_USERNAME \
  --env KAGGLE_KEY=YOUR_KEY \
  --tensorboard-dir /spell/tensorboards/model_4 \
  "chmod +x /spell/scripts/download_data.sh; chmod +x /spell/scripts/upgrade_env.sh; /spell/scripts/download_data.sh; /spell/scripts/upgrade_env.sh; python /spell/models/model_4.py"
```
