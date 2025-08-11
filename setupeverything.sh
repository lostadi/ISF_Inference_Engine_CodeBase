#!/bin/bash

# Script to detect NVIDIA GPU, install CUDA 12.8 toolkit, PyTorch with CUDA support,
# and compile Ollama optimized for the detected GPU compute capability on Ubuntu 22.04-based systems (e.g., elementary OS 7).

# Exit on error
set -e

# Step 1: Check for NVIDIA GPU
if ! lspci | grep -i nvidia; then
  echo "No NVIDIA GPU detected. Exiting."
  exit 1
fi

# Step 2: Install NVIDIA drivers if nvidia-smi is not available
if ! command -v nvidia-smi &> /dev/null; then
  echo "NVIDIA drivers not found. Installing latest drivers from PPA."
  sudo apt update
  sudo apt install software-properties-common -y
  sudo add-apt-repository ppa:graphics-drivers/ppa -y
  sudo apt update
  sudo apt install nvidia-driver-550 -y  # Recent driver supporting CUDA 12.8
  echo "Drivers installed. Please reboot your system and run this script again."
  exit 0
fi

# Step 3: Detect GPU compute capability (use the first GPU if multiple)
COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [ -z "$COMPUTE" ]; then
  echo "Failed to detect compute capability. Exiting."
  exit 1
fi
ARCH=${COMPUTE//./}
echo "Detected GPU compute capability: $COMPUTE (ARCH: $ARCH)"

# Check if compute capability is supported (minimum 5.0 for Ollama CUDA)
if (( $(echo "$COMPUTE < 5.0" | bc -l) )); then
  echo "GPU compute capability $COMPUTE is below 5.0. CUDA acceleration may not be supported. Proceeding anyway."
fi

# Step 4: Install CUDA Toolkit 12.8
echo "Installing CUDA Toolkit 12.8..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-8 -y

# Step 5: Set CUDA environment variables and add to .bashrc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
if ! nvcc --version; then
  echo "CUDA installation failed. Exiting."
  exit 1
fi
echo "CUDA 12.8 installed successfully."

# Step 6: Install PyTorch with CUDA 12.8 support
echo "Installing PyTorch with CUDA 12.8 support..."
sudo apt install python3-pip -y
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch CUDA
python3 -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
if [ $? -ne 0 ]; then
  echo "PyTorch installation or CUDA check failed."
fi

# Step 7: Compile Ollama from source optimized for the detected GPU
echo "Compiling Ollama from source with CUDA optimization..."
sudo apt install golang-go cmake git build-essential -y
git clone https://github.com/ollama/ollama.git
cd ollama
git submodule update --init --recursive
export CMAKE_CUDA_ARCHITECTURES=$ARCH
go generate ./...
go build .
echo "Ollama compiled successfully."

# Step 8: Install Ollama binary (optional: move to /usr/local/bin)
sudo cp ollama /usr/local/bin/ollama
echo "Ollama binary copied to /usr/local/bin. You can run 'ollama serve' to start the server."

# Cleanup
cd ..
# rm -rf ollama  # Uncomment to remove source after build

echo "Setup complete! Open a new terminal or source ~/.bashrc. Test Ollama with 'ollama run llama3' (it will download the model)."
echo "Note: For multi-GPU systems, set CUDA_VISIBLE_DEVICES to select specific GPUs if needed."
