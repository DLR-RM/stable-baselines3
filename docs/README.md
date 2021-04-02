## Stable Baselines3 Documentation

This folder contains documentation for the RL baselines.


### Build the Documentation

#### Install Sphinx and Theme

```
pip install sphinx sphinx-autobuild sphinx-rtd-theme sphinx-autodoc-typehints 
```

#### Building the Docs

In the `docs/` folder:
```
make html
```

if you want to building each time a file is changed:

```
sphinx-autobuild . _build/html
```
