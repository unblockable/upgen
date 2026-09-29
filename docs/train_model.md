# Model Training

To train the model, follow the following steps:
We assume that installation is done and you are in the root directory

1. First go to greeting directory

```
[~/upgen] $ cd src/greeting/
```

2. In next step, generate the input file (write the script in shell to create) and place the input file in the current directory (greeting/).
We assume that the name of input file is `input_file.txt` for this documentation, but you can keep any name.

3. Then, find the best parameters for model. Use the following command:

```
[~/upgen/src/greeting] $ python3 train.py -c 0 -y input_file.txt --output_filepath best_params.pkl
```

Note that if you don't have a CUDA GPU, then remove the -c 0 flag. This is only for CUDA/NVIDIA GPUs.
This step will generate a file called `best_params.pkl`

It may take some time to create it, so be patient.

4. After previous step, create a directory inside the greeting/, we name it trained_model/ but users can pick any name according to their own.

```
[~/upgen/src/greeting] $ mkdir trained_model/
```

5. Then train the model

```
[~/upgen/src/greeting] $ python3 train.py -c 0 -b best_params.pkl -d trained_model/ input_file.txt
```

It will create the model artifacts inside trained_model/ directory.
To check sample model artifacts, please check the [model_samples](model_samples/) directory.
This completes the model training and now we are ready to generate the PSFs.

---
