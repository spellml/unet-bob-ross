#!/bin/bash
set -v
kaggle datasets download residentmario/segmented-bob-ross-images
unzip  -qq segmented-bob-ross-images.zip -d /mnt/segmented-bob-ross-images/
rm segmented-bob-ross-images.zip
echo "Done downloading data."