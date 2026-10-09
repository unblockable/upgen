# PSF generation

Before generating a PSF, obtain a greeting string model by following the
instructions in the [greeting string model guide](greeting_model.md).

Run the generator from the repository root:

```
$ make generate
```

This reads the model artifacts under `build/` and creates `build/upgen.psf`.

The packaged `config.json` is used by default, *but you are free to
change/randomize the parameters to create various protocol distributions.*. The
included `config.json` file just sets all random choices as equally likely.

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
