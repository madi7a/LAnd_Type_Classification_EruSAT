# 🌍 Land Type Classification from Satellite Images

This project focuses on classifying different types of land cover (e.g., forest, agriculture, water) from satellite imagery using deep learning techniques. The goal is to assist researchers, developers, and environmental analysts in monitoring land use and land cover changes effectively through automated AI systems.

🔗 **Live Demo**: [Try the App on Streamlit](https://landtypeclassification-erusat.streamlit.app/)

---

## 🎯 Project Objective

Classifying satellite imagery can provide critical insights into how land is used, how it changes over time, and what kind of human or natural activity is occurring. Manual classification is time-consuming and inconsistent, so our aim was to:

- Build a robust image classification model using deep learning
- Compare multiple CNN architectures
- Select the best-performing model based on evaluation metrics
- Deploy the model as an interactive web app

---

## 🧠 Models Compared

We experimented with three different deep convolutional neural networks:

| Model         | Train Acc | Val Acc | Test Acc | Precision | Recall | F1 Score |
|---------------|-----------|---------|----------|-----------|--------|----------|
| **ResNet18**  | 99.74%    | 98.15%  | **98.75%**  | 98.88%    | 98.75% | **98.80%**  |
| VGG16         | 99.25%    | 96.57%  | 96.25%  | 96.30%    | 96.25% | 96.20%   |
| AlexNet       | 98.05%    | 94.10%  | 93.85%  | 94.02%    | 93.85% | 93.70%   |

### 🔍 Evaluation

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Epochs**: 25
- **Dataset Split**: 80% Train / 10% Val / 10% Test

We found **ResNet18** to be the most robust model, achieving the highest generalization on unseen data.

---

## 📊 Confusion Matrix (ResNet18)

![Confusion Matrix](confusion_matrix_resnet18.png)

- The model shows strong class-wise accuracy across all categories.
- No significant class imbalance or confusion.

---

## 🛠️ Tech Stack

- **Framework**: PyTorch
- **UI/Deployment**: Streamlit
- **Language**: Python
- **Model**: ResNet18 with ImageNet pre-trained weights

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/your-username/landtype-classification.git
cd landtype-classification
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

The trained ResNet18 model was deployed using **Streamlit Cloud**. Users can upload their own satellite image and get real-time classification results with a simple UI.

🔗 [Deployed App](https://landtypeclassification-erusat.streamlit.app/)

---

## 📁 Dataset

- RGB satellite images (224x224)
- Classes: Agriculture, Forest, Residential, River, Water, and more
- Dataset is preprocessed (resized, normalized) and augmented (rotation, flipping)

> You can modify the notebook to train on your own dataset.

---

## 📌 Key Highlights

✅ Compared 3 CNN architectures with full metrics  
✅ Highest accuracy: **ResNet18 (98.75%)**  
✅ Deployed real-time prediction web app  
✅ Excellent generalization across multiple land classes

---

## 📃 License

This project is licensed under the MIT License.


