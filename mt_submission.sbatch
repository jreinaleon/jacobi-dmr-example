#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH -N 5
#SBATCH -o dmr_exec.out

export MPICH_PATH=/apps/dmr
echo "MPICH-3.2 installed in $MPICH_PATH"
echo "DMR installed with Slurm in $DMR_PATH"
echo "Slurm installed in $SLURM_ROOT"

SLURM_BIN=$SLURM_ROOT/bin
SLURM_SBIN=$SLURM_ROOT/sbin
SLURM_CONF_DIR=$DMR_PATH/slurm-confdir
SLURM_CONF_FILE=$SLURM_CONF_DIR/slurm.conf

i=0;
nodes=""
coma=","

rm -rf $SLURM_CONF_DIR
cp -r $DMR_PATH/slurm-spawn/confdir $SLURM_CONF_DIR
mv $SLURM_CONF_DIR/slurm.conf.base $SLURM_CONF_FILE
echo "SlurmUser = $USER" >> $SLURM_CONF_FILE
echo "SlurmdUser = $USER" >> $SLURM_CONF_FILE
echo "JobCredentialPrivateKey = $SLURM_CONF_DIR/slurm.key" >> $SLURM_CONF_FILE
echo "JobCredentialPublicCertificate = $SLURM_CONF_DIR/slurm.cert" >> $SLURM_CONF_FILE
echo "StateSaveLocation = $SLURM_ROOT/var" >> $SLURM_CONF_FILE
echo "SlurmdSpoolDir = $SLURM_ROOT/var/slurmd.%n " >> $SLURM_CONF_FILE
echo "SlurmctldPidFile = $SLURM_ROOT/var/slurmctld.pid " >> $SLURM_CONF_FILE
echo "SlurmdPidFile = $SLURM_ROOT/var/slurmd.%n.pid " >> $SLURM_CONF_FILE
echo "SlurmctldLogFile = $SLURM_ROOT/var/slurmctld.log" >> $SLURM_CONF_FILE
echo "SlurmdLogFile = $SLURM_ROOT/var/slurmd.%n.log" >> $SLURM_CONF_FILE
echo "AccountingStorageLoc = $SLURM_ROOT/var/accounting" >> $SLURM_CONF_FILE
echo "JobCompLoc = $SLURM_ROOT/var/job_completions" >> $SLURM_CONF_FILE
rm -rf $SLURM_ROOT/var
mkdir -p $SLURM_ROOT/var
rm -rf $SLURM_ROOT/lock/resize
echo "" > hostfile.txt
echo "" >  $SLURM_ROOT/var/accounting

NODELIST="$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d -s)"
for node in $NODELIST; do
	#echo "$node - $(hostname)"
	if [ "$node" == "$(hostname)" ]; then
		echo "ControlMachine=$(hostname)" >> $SLURM_CONF_FILE
		#echo "Controller $(hostname)"
	else
		echo $node >> hostfile.txt
		echo "NodeName=$node CPUs=12 CoresPerSocket=6 ThreadsPerCore=1 State=Idle Port=7009" >> $SLURM_CONF_FILE
		nodes=$node$coma$nodes
	fi
done;

echo "PartitionName=dmrTest Nodes=$(echo $nodes | sed 's/.$//') Default=YES MaxTime=INFINITE State=UP" >> $SLURM_CONF_FILE

NNODES=$(($SLURM_NNODES-1))
NODELIST="$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)"

$SLURM_SBIN/slurmctld -cDv  &
mpiexec -n $NNODES --hostfile hostfile.txt $SLURM_SBIN/slurmd -cDv &
$SLURM_BIN/sinfo

$SLURM_BIN/sbatch -Jdmr_jacobi -N1 launch.sh &

$SLURM_BIN/sinfo

aux=$( $SLURM_BIN/squeue | wc -l );
while [ $aux -gt 1 ]; do
	aux=$( $SLURM_BIN/squeue | wc -l );
	echo "$aux jobs remaining...";
	#sleep 10;
done
#sleep 10
echo "Finishing...";
