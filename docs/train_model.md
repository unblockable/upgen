# Model Training

Before creating a PSF, we first need a model to generate greeting strings. To
train the model, follow the following steps.

1. First go to greeting directory

```
$ cd src/greeting/
```

2. Next, generate the input file and place it in the current directory
   (`greeting/`). The included download script retrieves repository names from
   one hour of [GH Archive](https://www.gharchive.org/) data:

```
$ python3 ../../scripts/download_gharchive_repos.py
```

By default, the script downloads the archive for January 1, 2015 at 15:00 UTC
and writes up to 10,000 names to `repos.txt`. Run it with `--help` to see
options for changing the date, hour, limit, or output file. It uses only the
Python standard library. You can also use your own script, provided that it
writes one `owner/repository` entry per line.

3. Then, find the best parameters for the model. Use the following command:

```
$ python3 train.py -c 0 -y repos.txt --output_filepath best_params.pkl
```

Note that if you don't have a CUDA GPU, then remove the -c 0 flag. This is only
for CUDA/NVIDIA GPUs. This step will generate a file called `best_params.pkl`

This command may take a while to complete.

4. After previous step, create a directory inside greeting/. We name it
   trained_model/, but you can pick any name.

```
$ mkdir trained_model/
```

5. Then train the model

```
$ python3 train.py -c 0 -b best_params.pkl -d trained_model/ repos.txt
```

It will create the model artifacts inside trained_model/ directory. This completes model training and now we are ready to generate PSFs.
