#!/bin/bash
set -euxo pipefail

trap 'echo "ERROR: init.sh failed at line $LINENO"' ERR

cd /home/onyxia/work

echo "USER=$(whoami)"
echo "HOME=$HOME"
echo "PWD=$(pwd)"

### === Install Miniforge (user-local) ===
MINIFORGE=Miniforge3-Linux-x86_64.sh
INSTALL_DIR=$HOME/miniforge3

echo "🔧 Installing Miniforge..."
wget https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE -O $MINIFORGE
bash $MINIFORGE -b -p $INSTALL_DIR
rm $MINIFORGE

# Enable conda commands in this shell
source $INSTALL_DIR/etc/profile.d/conda.sh

### === Create and activate environment ===
echo "🧪 Creating conda environment 'foccus_tolosa'..."
conda create -y -n foccus_tolosa python=3.12.13
conda activate foccus_tolosa

# Install mamba
conda install -y -c conda-forge mamba

### === Install exact packages ===
echo "📦 Installing required packages..."
mamba install -y -c conda-forge \
  json==0.15.0 \
  pandas==2.2.3 \
  numpy==2.5.0 \
  matplotlib==3.11.0 \
  geopandas \
  cartopy==0.25.0 \
  xarray==2026.4.0 \
  cmocean==4.0.3 \
  IPython==9.15.0 \
  netcdf4==1.7.4 \
  zarr==3.2.1 \
  ipywidgets==8.0.0 \
  Shapely==2.1.2 \
  ipykernel jupyter nbformat nbconvert s3fs

### === Register kernel for Jupyter ===
echo "🔗 Registering Jupyter kernel..."
python -m ipykernel install --user --name foccus_tolosa --display-name "Python (foccus_tolosa)"

### === Download notebook and helper script ===
echo "📥 Downloading notebook and script..."
wget -N https://github.com/fbrosse/FOCCUS_storm_surge_Tolosa_demonstrator/raw/main/main.ipynb
wget -N https://github.com/fbrosse/FOCCUS_storm_surge_Tolosa_demonstrator/raw/main/foccus_helpers.py
wget -N https://github.com/fbrosse/FOCCUS_storm_surge_Tolosa_demonstrator/raw/main/download_from_s3.py

### === Embed kernel metadata ===
echo "⚙️ Embedding kernel metadata into notebook..."
python  - <<EOF
import nbformat

nb_path = "main.ipynb"
nb = nbformat.read(open(nb_path), as_version=nbformat.NO_CONVERT)

nb["metadata"]["kernelspec"] = {
    "name": "foccus_tolosa",
    "display_name": "Python (foccus_tolosa)",
    "language": "python"
}

nbformat.write(nb, open(nb_path, "w"))
EOF

### === Clear notebook output ===
echo "🧼 Clearing cell outputs..."
jupyter nbconvert --clear-output --inplace main.ipynb

echo "✅ Setup complete. You can now open main.ipynb and it will use the 'foccus_tolosa' kernel by default."

### === Download input ===

# Dataflow
wget -N https://github.com/fbrosse/FOCCUS_storm_surge_Tolosa_demonstrator/raw/main/figs/D321.png

# Logos
mkdir -p logos
cd logos

# Base path to raw files on GitHub
BASE_URL="https://github.com/fbrosse/FOCCUS_storm_surge_Tolosa_demonstrator/raw/main/logos"

# List of files to download
FILES=(
 shom.png
 moi.png
 foccus.png
)

# Download each file
for file in "${FILES[@]}"; do
  echo "Downloading $file..."
  wget -nc "$BASE_URL/$file"
done

cd ..
