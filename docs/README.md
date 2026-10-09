# Documentation

To generate a PSF:

1. Follow the [installation guide](install.md).
2. Obtain the required greeting-model artifacts. Downloading a pretrained model
   will be the recommended path; training your own is optional. See the
   [greeting string model guide](greeting_model.md).
3. Follow the [PSF generation guide](generate_psf.md).

Additional references:

- [Optional model training](train_model.md)
- [Command-line reference](cli.md)

The Make-based workflow stores downloaded data, trained models, and generated
PSFs under the ignored `build/` directory.
