#  Image Classification with GAN-Based Data Augmentation (Cow vs Horse)

How much can synthetic images from DCGAN and a FastGAN-inspired model help a CNN trained from scratch on only **41 images per class**? This project also trains a CycleGAN for unpaired Cow <-> Horse translation.

*Master 1 Computer Vision project, Université Paris-Saclay, 2026.*

## Setup

- **Task:** binary classification, Cow vs Horse
- **Data:** 41 training images and 41 test images per class (82 + 82). The data comes from a university course and is not redistributed in this repository.
- **Classifier:** small CNN built from scratch (no pretrained weights)
- **Test set:** 82 images, used only for final evaluation

## Methods

1. **Baseline CNN** trained on the real images only (128x128).
2. **DCGAN** (one generator per class, 800 epochs, 128x128), 100 generated images per class added to the training set.
3. **Selective DCGAN augmentation:** 12 generated Horse images chosen by visual inspection and added to the harder class; CNN with BatchNorm, GAP, flip/rotation augmentation at 256x256.
4. **FastGAN-inspired generator** (skip-layer excitation, differentiable augmentation, hinge loss, EMA generator), 20 selected images per class added.
5. **CycleGAN** (ResNet generators, instance norm, cycle + identity losses) for unpaired Cow <-> Horse translation.

## Results

Accuracy on the 82-image test set:

| Training set | Test accuracy |
|---|---:|
| Baseline CNN, real images only | 51.2 % (predicts almost every image as Cow) |
| CNN + all DCGAN images (100 per class) | 57.3 % |
| CNN + 12 selected DCGAN Horse images (seed 23) | 65.9 % |
| CNN + 20 selected FastGAN-inspired images per class | 63.4 % (seed 23), 64.6 % (unseeded run) |

![FastGAN samples](results/fastgan_samples_horse.png)

Confusion matrices and generated samples are in [`results/`](results/).

## Important caveats

- **The test set is small (82 images).** A 95 % confidence interval on an accuracy of about 65 % is roughly +/- 10 points, so the gaps between the augmented models are not statistically conclusive.
- **Seed variance is large.** For the selective DCGAN setup, 10 seeds gave accuracies between 45 % and 67 %, with a mean of 51.7 %. The 65.9 % result uses the best seed, which was chosen using test accuracy, so it is optimistic.
- **GAN images were selected by eye**, which is subjective and not reproducible without the same index lists.
- **Validation accuracy in the augmented experiments is not meaningful**, because the random validation split mixes real and generated images derived from the training set.
- No image-quality metric (FID/KID) was computed for the generated images.

## Repository structure

```
notebooks/   main notebook (Google Colab, TensorFlow/Keras)
results/     confusion matrices, training curves, generated samples, CycleGAN examples
data/        see data/README.md
```

## How to run

```bash
git clone https://github.com/YOUR_USERNAME/gan-augmentation-cow-vs-horse.git
cd gan-augmentation-cow-vs-horse
pip install -r requirements.txt
jupyter notebook notebooks/cow_vs_horse_gan_augmentation.ipynb
```

The notebook was developed on Google Colab with the data on Google Drive; set `base_dir` in the first cells to your own copy of the dataset (`Train/Cow`, `Train/Horse`, `Test/Shape1`, `Test/Shape2`).

## Tools

Python, TensorFlow/Keras, NumPy, scikit-learn, Matplotlib, Seaborn.
