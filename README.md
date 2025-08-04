
#  Skin Disease Classification with ViT

This project is a deep learning-based image classification system that uses the **Vision Transformer (ViT)** to identify 8 different skin diseases from images.

---

##  Dataset

* Dataset structure:

  ```
  skin-disease-dataset/
    ├── train_set/
    │   ├── FU-nail-fungus/
    │   ├── FU-ringworm/
    │   ├── ...
    └── test_set/
        ├── FU-nail-fungus/
        ├── FU-ringworm/
        ├── ...
  ```
* Total images: `925` for training, `234` for testing
* 8 disease classes

---

##  Model & Training

* **Model**: [`ViT-B/16`](https://huggingface.co/google/vit-base-patch16-224-in21k) (`vit-base-patch16-224-in21k`)
* **Loss function**: CrossEntropyLoss
* **Optimizer**: Adam (`lr=5e-5`)
* **Epochs**: 10
* **Input size**: 224×224
* **Transform**: Resize → ToTensor

---

##  Results

* **Final Train Loss**: `0.0779`
* **Validation Accuracy**: `97.86%`
* **Validation Loss**: `0.1620`

---



##  Dependencies

* `torch`
* `torchvision`
* `transformers`
* `matplotlib`
* `PIL`
* `pandas`


