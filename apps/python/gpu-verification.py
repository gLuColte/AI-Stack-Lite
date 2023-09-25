# importing torch  
import torch  

# Get index of currently selected device  
print(f"Current GPU Device: {torch.cuda.current_device()}") 

# get number of GPUs available  
print(f"GPU Device Count: {torch.cuda.device_count()}") 

# get the name of the device  
print(f"GPU Device 0 Name: {torch.cuda.get_device_name(0)}")