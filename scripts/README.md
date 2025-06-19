# Example scripts to run/submit jobs for miso

* `run_miso.py`: Example run script. Takes anndata with modality feature stored in the obsm and calculates the niches

* `submit_miso.sh`: Example batch script
  * run with `$sbatch submit_miso.sh log_name [args]`, where `[args]` are the arguments for the `run_miso.py` script. Saves log as `out_miso_log_name.out`
