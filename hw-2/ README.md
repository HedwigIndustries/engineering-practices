# engineering-practices-hw-2

### Install git-lfs:

```bash
git lfs install
git lfs track '*.csv' # track .csv format
git lfs track <dataset_name> # track dataset file
git lfs untrack <dataset_name> # untrack dataset file
git add .gitattributes
```

After the last command, the ```.gitattributes``` file will be created, just setting up this file allows you to track
different files.

Add `dataset.csv` - the dataset contains characteristics of processors, the dataset was parsed from the
site https://www.chaynikam.info/cpu_table.html by `BeautifulSoap` library.

### View tracked files:

```bash
git lfs ls-files
```

### Start tracking dataset files:

```bash
git clone <rep_url | rep_ssh_key>
git lfs pull
```

### Update or append dataset files:

```bash
git add <path_to_files>
git commit -m <message>
git push
```

```path_to_files``` is responsible for the path of changing file or to the new tracked file.

### Roll back to previous version:

```bash
git reset --hard <commit_head_id>
```

P.S. For example, I created `update_dataset.csv` file to show how my dataset changes.
If I overwrite `dataset.csv` in the same file, then git lfs will allow me to track the dataset and revert to a previous
version of it if I wish.
