# PSF Generation

After training the model, follow the following instructions to generate a PSF:

1. Go back to src/ directory

```
[~/upgen/src/greeting] $ cd ..
```

2. Use generate.py to create PSFs

```
[~/upgen/src] $ python generate.py \
  config.json \
  greeting/best_params.pkl \
  greeting/trained_model/encoder.pkl \
  greeting/trained_model/model.torch \
  -n 1 \
  -o name.psf
```

This will create a PSF file in the src/ directory.

To create multiple PSFs, you can run bash scripts. 
For example:

```
[~/upgen/src] $ for i in {1..10}
do
    python generate.py \
        config.json \
        greeting/best_params.pkl \
        greeting/trained_model/encoder.pkl \
        greeting/trained_model/model.torch \
        -n 1 \
        -o "psf_${i}.psf"
done
```

This script will create 10 PSFs. Users can customise it or use their own scripts to create required number of PSFs easily.

---
