#!/bin/bash
#SBATCH -A [ACCOUNT_NAME]
#SBATCH --partition=[PARTITION_NAME]
#SBATCH --mem=40G
#SBATCH --array=1-[MAX_JOBS_LIMIT]%50
#SBATCH --time=5000
#SBATCH --constraint="ARCH:X86&CPU_PROD:XEON"
#SBATCH --job-name=benchmark_memory_moderate
#SBATCH --cpus-per-task=4

#SBATCH --mail-user=[E-MAIL-address]
#SBATCH --mail-type=ALL
#SBATCH --output=[PATH_TO_OUTPUT]
#SBATCH --error=[PATH_TO_ERROR]

line=$(sed -n ${SLURM_ARRAY_TASK_ID}p < ./[LOCAL_PATH_TO_PARAMETERS_CSV_FILE])
rec_column1=$(cut -d',' -f1 <<< "$line")
rec_column2=$(cut -d',' -f2 <<< "$line")
if [ ! -f "$rec_column1" ]; then
  echo python3 $rec_column2
  srun --container-image=[PATH_TO_CONTAINER] --container-name=[CACHED_CONTAINER_NAME] --container-mounts=[MOUNTS] --container-workdir=[WORK_DIR] python3 $rec_column2
fi
