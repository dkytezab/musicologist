import kagglehub

# Download latest version
path = kagglehub.dataset_download("googleai/musiccaps")

print("Path to dataset files:", path)
