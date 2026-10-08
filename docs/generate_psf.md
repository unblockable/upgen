# PSF Generation

After training the model, follow the following instructions to generate a PSF:

1. Go back to src/ directory

```
$ cd ..
```

2. Use generate.py to create PSFs

```
$ python generate.py \
  config.json \
  greeting/best_params.pkl \
  greeting/trained_model/encoder.pkl \
  greeting/trained_model/model.torch \
  -n 1 \
  -o name.psf
```

This will create a PSF file in the src/ directory.

To create multiple PSFs, you can adjust the `-n` command line parameter.
