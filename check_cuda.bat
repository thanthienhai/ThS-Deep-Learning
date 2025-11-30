@echo off
"C:\Users\Than Thien\miniconda3\envs\ThS-Deep-Learning\python.exe" -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" > cuda_status.txt 2>&1
