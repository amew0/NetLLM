# For Pytorch


# cuda version
cu=124

# pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu${cu}
pip install torch==2.4.0
# new ones
pip install yacs
pip install sentencepiece
pip install protobuf

# For PyG
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.4.0+cu${cu}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.4.0+cu${cu}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.4.0+cu${cu}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.4.0+cu${cu}.html
pip install torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple torch_geometric

# For Gymnasium
conda install swig -y
pip install "Gymnasium[all]" # install without [all] if fails

# For other packages
#!!!!! IMPORTANT !!!!!
# get their versions from the readme.md
pip install numpy
pip install transformers
pip install munch
pip install openprompt
pip install peft