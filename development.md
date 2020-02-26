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
cd scripts/formulas/conda/eddl
conda build .
```