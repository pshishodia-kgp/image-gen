# A100 setup
## NVIDIA driver installation
Setup [docs](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)
1. sudo apt update && sudo apt install -y ubuntu-drivers-common
2. sudo ubuntu-drivers install
3. MOK Code - Need to generate a new MOK code after installing nvidia drivers. This code is asked during secure boot up (which is enabled by default in azure VM). This code gets entered from UEFI interface during secure boot up (enabled by default in azure). UEFI interface is not visible via SSH, disabled secure boot mode in azure for now.
## Cuda keyring installation
1. wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
2. sudo apt install -y ./cuda-keyring_1.1-1_all.deb
3. sudo apt update
4. sudo apt -y install cuda-toolkit-12-5
5. sudo reboot
6. nvidia-smi # To verify

Docs say to periodically upgrade NVIDIA drivers-
1. sudo apt update
2. sudo apt full-upgrade
