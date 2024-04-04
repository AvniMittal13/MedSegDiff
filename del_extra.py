import os

# Define the directory containing the files
directory = "/scratch/avinim.scee.iitmandi/MedSegDiff/results/ISIC_normal_var"

# Loop through files in the directory
for filename in os.listdir(directory):
    # Check if the filename matches the pattern "prefix_<num>"
    # /scratch/avinim.scee.iitmandi/MedSegDiff/results/ISIC/emasavedmodel_0.9999_368100.pt
    # /scratch/avinim.scee.iitmandi/MedSegDiff/results/ISIC/optsavedmodel052600.pt
    if filename.startswith("emasavedmodel_0.9999_") :
        try:
            # Extract the number part
            num = int(filename.split("_")[-1].split(".")[0])
            # Check if the number is within the range 0-2000
            if num >= 0 and num <= 243100:
                # Construct the full path to the file
                filepath = os.path.join(directory, filename)
                # Delete the file
                os.remove(filepath)
                print(f"Deleted file: {filepath}")
        except ValueError:
            # If the number part cannot be converted to an integer, skip the file
            pass
