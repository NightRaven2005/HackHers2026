import kagglehub
import os
import shutil

datasetdir = os.path.join(os.getcwd(), "datasets")
if not os.path.exists(datasetdir):
    os.makedirs(datasetdir)

path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
shutil.copytree(path, datasetdir, dirs_exist_ok=True)
print(f"Dataset downloaded to: {datasetdir}")