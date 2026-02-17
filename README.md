Looking through LIGO data for signs of black hole axion superradiance.

To use this code, checkout the `main` branch and then use `zenodo-get` to put the samples you want into a folder named `gwtc3_pe_samples`. Using NERSC to analyze GWTC-3, this looks like the following.

```
# Clone and enter the repository
git checkout main
cd ~/BH_superradiance

# Set up the environment
module add conda
conda activate [your_env_name]
pip install zenodo-get  # Note: usually pip installed, or 'conda install -c conda-forge'

# Download GWTC-3 PE samples into the target directory
mkdir -p gwtc3_pe_samples
cd gwtc3_pe_samples
zenodo_get 5546663
```
