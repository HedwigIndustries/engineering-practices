# engineering-practices-hw-3

### Install dvc:

```bash
sudo snap install dvc --classic # switch to wsl and download dvc
dvc --version # 3.36.1 correctly installed
```

### Commands dvc:

```bash
dvc init --subdir # This will create a .dvc file and configure the DVC for this project.
dvc add <file> # This will create a .dvc file that will track this file.
dvc import <URL> # To load data from a remote source, such as the cloud or a code repository.
dvc repro # To play back all files from your DVC configuration file to ensure they are up to date.
dvc remote add <имя> <URL> # Connecting remote storage
dvc push # To send data and models to remote storage to store versions of your data and model results.
```

### Run pipeline:

So, let's run our Python code, where we divide the dataset into a train, build a prediction using the `Prophet` model,
calculate the error using `MAE/MAPE` metrics.

Then we add new `.csv` files and add this with `dvc`.

P.S. I couldn't understand how `run` command works in `dvc`.
