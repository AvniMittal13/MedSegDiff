import torch

# Load the model
model_state_dict = torch.load("/scratch/avinim.scee.iitmandi/MedSegDiff/results/ISIC_normal_actual_256_100time_linear/emasavedmodel_0.9999_001000.pt", map_location=torch.device('cpu'))

# Calculate the total number of parameters
total_params = sum(p.numel() for p in model_state_dict.values())
print("Total number of parameters: ", total_params)
