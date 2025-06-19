# Xenium single cell miso

This repo was forked from the original [miso repository](https://github.com/kpcoleman/miso) and modified to accomodate large xenium datasets with batching and cagra knn to calculate adjacency matrices, with several other changes. This version does not use the HIPT model for H&E images and instead takes as input pre-calculated foundation model embeddings that have already been associated to cells (see [here](https://github.com/Tyler-Lam/Knottlab-Tutorials/tree/main/foundation_models) for an example implementation).

### Setup:
```
mamba create -n miso_rapids python=3.11.11
mamba activate miso_rapids
mamba install -c rapidsai -c conda-forge -c nvidia cuvs cuda-version=12.8
mamba install -c conda-forge ipykernel ipywidgets
git clone https://github.com/Tyler-Lam/miso.git
cd miso
python -m pip install .
```

## Software Requirements

python==3.11.11  
einops==0.8.1  
numpy==2.2.6  
opencv-python==4.10.0  
Pillow==11.2.1  
scanpy==1.11.2  
scikit-image==0.25.2  
scikit-learn==1.7.0  
scipy==1.15.2  
setuptools==75.8.2  
torch==2.4.0  
torchvision==0.19.0  
tqdm==4.67.1  
