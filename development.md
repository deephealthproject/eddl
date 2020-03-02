# Development

## Requirements

Use the conda environment, or install these additional packages:

```
sudo apt-get install -y doxygen
pip install sphinx
pip install sphinx_rtd_theme
pip install breathe
```

## Documentation

### Doxygen

```
cd docs/doxygen
doxygen
```

### Sphinx

```
cd docs/source
make clean; make html
```

## Badges tool

Generated using: [shields.io](https://shields.io/)


## Build conda package

```
cd formulas/conda/eddl
conda build .

# Test installation
conda activate
conda install --use-local eddl

# Test cpp example
# rm -rf ./*; cmake ..; make; ./main
cmake ..; make; ./main

# Convert to platforms
conda convert --platform all ... -o ~/precompiled

# Upload
anaconda upload ~/anaconda3/conda-bld/...

# Upload all
find ~/precompiled -name "*.bz2" -type f -exec anaconda upload "{}" \;
```