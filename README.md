
# Skin Disease Classification using ViT (Vision Transformer)

This project uses a fine-tuned Vision Transformer (ViT) model from Hugging Face to classify images of skin diseases into 8 categories. The dataset includes images of various infectious skin diseases.

## 📁 Dataset

The dataset should be structured as follows:

```
skin-disease-dataset/
│
├── train_set/
│   ├── FU-nail-fungus/
│   ├── FU-ringworm/
│   ├── VI-shingles/
│   ├── BA-impetigo/
│   ├── FU-athlete-foot/
│   ├── VI-chickenpox/
│   ├── PA-cutaneous-larva-migrans/
│   └── BA- cellulitis/
│
└── test_set/
    └── (same structure as train_set)
```

Make sure the dataset path is set correctly in the script:

```python
path="/kaggle/input/skin-disease-dataset/skin-disease-datasaet"
```

---

## 🔧 Requirements

Install dependencies with:

```bash
pip install torch torchvision transformers matplotlib pandas
```

---

## 🚀 Training the Model

The model used is `google/vit-base-patch16-224-in21k` from Hugging Face Transformers.

### Key Components:

* **Model:** Vision Transformer (ViT)
* **Loss:** Cross Entropy
* **Optimizer:** Adam
* **Epochs:** 10
* **Device:** GPU (CUDA) if available

### Run Training:

The script will:

* Load and preprocess the dataset
* Define a custom PyTorch dataset class
* Train the model for 10 epochs
* Save the trained model as `vit_infections.pth`

---

## 📊 Results

* **Final Training Loss:** \~0.078
* **Validation Accuracy:** **97.86%**
* **Validation Loss:** 0.162

---

## 🧪 Inference

After training, use the model for inference:

```python
model.load_state_dict(torch.load("vit_infections.pth"))
model.eval()
```

---

## 💾 Saving the Model

```python
torch.save(model.state_dict(), "vit_infections.pth")
```

---

## 📌 Notes

* The model layers `classifier.bias` and `classifier.weight` were randomly initialized and trained from scratch since the number of classes differs from the pre-trained model's original setting.
* Make sure your GPU has enough memory (\~12GB recommended) or reduce the batch size accordingly.

---

## 🧠 Classes Mapping

```python
{'FU-nail-fungus': 0, 'FU-ringworm': 1, 'VI-shingles': 2, 'BA-impetigo': 3,
 'FU-athlete-foot': 4, 'VI-chickenpox': 5, 'PA-cutaneous-larva-migrans': 6,
 'BA- cellulitis': 7}
```


