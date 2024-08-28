# Set up OpenGL - necessary?
sudo apt-get update -y
sudo apt-get install -y cmake pkg-config
sudo apt-get install -y mesa-utils libglu1-mesa-dev freeglut3-dev mesa-common-dev
sudo apt-get install -y libglew-dev libglfw3-dev libglm-dev
sudo apt-get install -y libao-dev libmpg123-dev
sudo apt-get install -y python3-opengl
sudo apt-get install -y fontconfig
sudo apt install -y xvfb

# First get a miniconda type env
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
chmod +x Miniforge3-Linux-aarch64.sh
./Miniforge3-Linux-aarch64.sh

# create a conda env
conda create -n benchmarl python==3.11
conda activate benchmarl

# Install pytorch (CPU)
conda install pytorch torchvision cpuonly -c pytorc

# Install tensordict from source
cd
git clone git@github.com:pytorch/tensordict.git
cd tensordict
python setup.py develop

# Install torchrl from source
cd
git clone git@github.com:pytorch/rl.git
cd rl
python setup.py develop

cd
git clone git@github.com:robj0nes/BenchMARL.git
cd BenchMARL
git checkout dev
pip install -e .

rm -rf VectorizedMultiAgentSimulator
git clone git@github.com:robj0nes/VectorizedMultiAgentSimulator.git
cd VectorizedMultiAgentSimulator
git checkout goals_from_image
pip install -e .

# Install other deps.
pip install -y torch_geometric seaborn opencv-python