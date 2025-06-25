# 
# https://www.youtube.com/watch?v=Zykqd6lBa1M
# https://www.youtube.com/watch?v=WjUeA-ghhvo&pp=0gcJCR0AztywvtLA
# https://www.youtube.com/watch?v=3tHQZolOYeY
# https://www.youtube.com/watch?v=6O8f8J0ci80

# https://www.swig.org/download.html
# pip install setuptools wheel - Python needs C++ compilers to build extensions.



# Function QS
import torch
import gymnasium as gym

env = gym.make("LunarLander-v2")
env.reset()



OPTIMIZE_WITH_HARDWARE = False

device = torch.device('cpu')
if OPTIMIZE_WITH_HARDWARE:
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'Selected device: MPS (Metal Performance Shaders)')
    elif torch.backends.cuda.is_available():
        device = torch.device('cuda')
        print(f'Selected device: GPU with CUDA support')
else:   
    print(f'Selected device: CPU')