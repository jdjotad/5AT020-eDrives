# Load modules 
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

# Create the virtual environment (only once)
python3 -m venv .venv/5AT020

# Install ipykernel inside it (only once)
.venv/5AT020/bin/python -m pip install ipykernel

# Add it as a Jupyter kernel
.venv/5AT020/bin/python -m ipykernel install --user --name=5AT0020 --display-name "Python (.venv/5AT020)"

# Install the required packages
.venv/5AT020/bin/python -m pip install -r requirements.txt

# Download eDrives repository
git clone https://github.com/jdjotad/5AT020-eDrives ./temp

# Copy the content of the repository to the current directory
rsync -av --exclude='.git' --exclude='.gitignore' ./temp/ ./

# Remove the temporary repository
rm -rf ./temp
