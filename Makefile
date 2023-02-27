
.PHONY: create-env activate-env deactivate-env install-local


create-env:
    conda create --name music --file requirements.txt

# Activate conda environment
activate-env:
    conda activate music

# Deactivate conda environment
deactivate-env:
    conda deactivate


install-local:
    pip install -e .
