# PSF Generation

After training the model, follow the following instructions to generate a PSF:

Run the generator from the repository root:

```
$ make generate
```

This reads the model artifacts under `build/` and creates `build/upgen.psf`.
The directory is ignored by Git. The packaged `config.json` is used by default.
If the environment was installed with the CUDA dependency set, use
`make generate MODEL_EXTRA=model-cuda` to keep that set selected. Generation
itself does not require a GPU.

To create multiple PSFs or choose a different output path, override the Make
variables:

```
$ make generate NUM_GENERATED=5 PSF_FILE=build/examples.psf
```

Additional generator options can be passed through `ARGS`, for example
`make generate ARGS='--seed 12345'`.
