import torch

def load_model_parameters(model_path):
    # Load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Print the parameters and their shapes
    for name, param in model.items():
        print(f"Parameter name: {name}, Shape: {param.shape}")

# Replace 'model.pt' with the path to your .pt file
model_path = '/scratch/avinim.scee.iitmandi/MedSegDiff/results/ISIC_normal_actual_256_100time_linear/emasavedmodel_0.9999_001000.pt'
load_model_parameters(model_path)
