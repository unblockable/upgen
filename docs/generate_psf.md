# PSF generation

Before generating a PSF, obtain a greeting string model by following the
instructions in the [greeting string model guide](greeting_model.md).

Run the generator from the repository root:

```
$ make generate
```

This passes `assets/config.json` to the generator, reads the model artifacts
under `build/`, and creates `build/upgen.psf`.

The configuration is a generator input, so you can change it to create different
protocol distributions. The included `assets/config.json` sets all random
choices as equally likely. Select a different configuration with the
`CONFIG_FILE` Make variable:

```
$ make generate CONFIG_FILE=assets/my-config.json
```

If the environment was installed with the CUDA backend, use
`make generate CUDA=0`
to keep that backend selected and run the greeting string model on CUDA device
0.

To create multiple PSFs or choose a different output path, override the Make
variables:

```
$ make generate NUM_GENERATED=5 PSF_FILE=build/examples.psf
```

Additional generator options can be passed through `ARGS`, for example
`make generate ARGS='--seed 12345'`.
