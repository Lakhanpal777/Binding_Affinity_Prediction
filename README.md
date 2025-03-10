# 🔬 **Binding Affinity Prediction using Deep Learning**  

This repository provides a deep learning model to **predict binding affinity** between proteins and ligands using structural and molecular features. The model integrates **molecular graphs, ligand coordinates, and molecular descriptors** to provide accurate predictions.  

---

## 📌 **Project Overview**  

- **Dataset Used**: [PDBBind (v2018)](http://www.pdbbind.org.cn/)  
- **Model Type**: Deep Learning (Fully Connected Neural Network)  
- **Features Used**:  
  - **Molecular Graphs** 🧬 (Graph Neural Network)  
  - **Molecular Descriptors** 🧪 (1D & 2D Features)  
  - **Ligand & Pocket Coordinates** 🔬 (3D Features)  
- **Goal**: Predict the binding affinity between proteins and ligands.  

---

## 🚀 **Installation & Setup**  

### 🔹 **1. Clone the Repository**  
```sh
git clone https://github.com/Lakhanpal777/Binding_Affinity_Prediction.git
cd Binding_Affinity_Prediction


2️⃣ **Create a Virtual Environment**

python -m venv env  
source env/bin/activate  # For Mac/Linux  
env\Scripts\activate  # For Windows  

3️⃣ Install Dependencies

pip install -r requirements.txt
 
