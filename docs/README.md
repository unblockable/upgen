This directory contains the documentation files of UPGen.

---

## Generation Process

Follow the given instructions to generate a PSF:

1. First, follow the [installation](install.md) guide.

2. Second, generate and train the model, whose complete procedure is given in
   [train_model.md](train_model.md) document.

3. Third, after training the model, create the PSF using
   [generate_psf.md](generate_psf.md)

The Make-based workflow stores downloaded data, trained models, and generated
PSFs under the ignored `build/` directory.
