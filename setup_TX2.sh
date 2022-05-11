#!bin/bash

echo "Installing basic libs for project ..."
pip3 install numpy scipy

echo "Installing opencv-python"
pip3 install opencv-python

echo "Installing pycuda"
pip3 install pycuda

echo "Installing tensorflow-python lib for TX2 Jetson KIT ..."
echo "Installing system packages required by TensorFlow ..."
sudo apt-get update -y
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
echo "Upgrading pip3 ..."
pip3 install -U pip testresources setuptools==49.6.0 
echo "Installing the Python package dependencies ..."
pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0
echo "installing the latest version of TensorFlow compatible with JetPack 4.6."
pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
# refer: https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html