# 🔬 Binding Affinity Prediction using Deep Learning  

This repository provides a deep learning model to predict binding affinity between proteins and ligands using structural and molecular features. The model integrates molecular graphs, ligand coordinates, and molecular descriptors to provide accurate predictions.  

---

## 📌 Project Overview  

- **Dataset Used**: PDBBind (v2018)  
- **Model Type**: Deep Learning (Fully Connected Neural Network)  
- **Features Used**:  
  - 🧬 **Molecular Graphs** (Graph Neural Network)  
  - 🧪 **Molecular Descriptors** (1D & 2D Features)  
  - 🔬 **Ligand & Pocket Coordinates** (3D Features)  
- **Goal**: Predict the binding affinity between proteins and ligands.  

---

## 🚀 Installation & Setup  

### 1️⃣ Clone the Repository  
```sh
git clone https://github.com/Lakhanpal777/Binding_Affinity_Prediction.git  
cd Binding_Affinity_Prediction  
```

### 2️⃣ Create a Virtual Environment  
```sh
python -m venv env  
source env/bin/activate  # For Mac/Linux  
env\Scripts\activate  # For Windows  
```

### 3️⃣ Install Dependencies  
```sh
pip install -r requirements.txt  
```
This will install all necessary libraries to run the model.  

---

## 🛠️ **How to Use the Model?**  

### 1️⃣ Prepare Input Files  
-Dataset should contain **protein-ligand pairs** in:  
  - **Ligand Files**: `.mol2` or `.sdf` format  
  - **Pocket Files**: `.pdb` format  

### 2️⃣ Run Prediction  
Use the `predict.py` script to predict binding affinity for your dataset.  
```sh
python predict.py  
```
The predicted values will be saved in **`binding_affinity_predictions.csv`**.  

---
