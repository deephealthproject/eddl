# Development

## Documentation

1. Generate doxygen files

```
cd docs/doxygen
doxygen
```

2. Generate sphinx files

```
cd docs/source
make html
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
cmake ..; make; ./main

# Convert to platforms
conda convert --platform all ... -o ~/precompiled

# Upload
anaconda upload ~/anaconda3/conda-bld/...

# Upload all
find ~/precompiled -name "*.bz2" -type f -exec anaconda upload "{}" \;
```