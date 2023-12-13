#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --qos=debug
#SBATCH -N9
#SBATCH -o dmr-exec.out

echo "MPICH-3.2 installed in $(which mpicc)"
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
#rm -rf $SLURM_ROOT/var/slurm*
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
		echo "NodeName=$node CPUs=48 CoresPerSocket=24 ThreadsPerCore=1 State=Idle Port=7010" >> $SLURM_CONF_FILE
		nodes=$node$coma$nodes
	fi
done;

echo "PartitionName=dmrTest Nodes=$(echo $nodes | sed 's/.$//') Default=YES MaxTime=INFINITE State=UP" >> $SLURM_CONF_FILE

NNODES=$(($SLURM_NNODES-1))
NODELIST="$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)"
mpiexec -n $SLURM_NNODES --hosts=$NODELIST hostname


MYSLURM_ROOT=${SLURM_ROOT}
MYSLURM_USER=${USER}
MYSLURM_CONF_DIR=$SLURM_CONF_DIR

# Get system info ====================================================
MYSLURM_MASTER=$(hostname)                     # Master node
MYSLURM_IP=$(hostname -i)                      # Master ip
NODELIST=$(scontrol show hostname | paste -d" " -s)

REMOTE_LIST=(${NODELIST/"${MYSLURM_MASTER}"})  # List of remote nodes (removing master)
_SLURM_SLAVES=${REMOTE_LIST[*]}                # "node1 node2 node3"

MYSLURM_SLAVES=${_SLURM_SLAVES// /,}           # "node1,node2,node3"
MYSLURM_NSLAVES=${#REMOTE_LIST[@]}             # number of slaves

if ((MYSLURM_NSLAVES == 0)); then
	echo "Error: MYSLURM_NSLAVES is zero (are you in the login node?)" >&2
	return 1
fi

echo "# Generating wrapper: mywrapper.sh"
sed -e "s|@MYSLURM_MASTER@|${MYSLURM_MASTER}|g" \
	-e "s|@MYSLURM_USER@|${MYSLURM_USER}|g" \
	-e "s|@MYSLURM_ROOT@|${MYSLURM_ROOT}|g" \
	-e "s|@MYSLURM_CONF_DIR@|${MYSLURM_CONF_DIR}|g" \
	$DMR_PATH/mywrapper.sh.base > mywrapper.sh

chmod a+x mywrapper.sh

###Remove & for interactive use!!!
srun -n 9 mywrapper.sh &
###

sleep 5

export SLURM_CONF=$SLURM_CONF_FILE
$SLURM_BIN/sinfo

$SLURM_BIN/sbatch -JdmrJob1 -N4 launch.sh &

$SLURM_BIN/squeue
$SLURM_BIN/sinfo

aux=$( $SLURM_BIN/squeue | wc -l );
while [ $aux -gt 1 ]; do
	aux=$( $SLURM_BIN/squeue | wc -l );
	echo "$aux jobs remaining...";
	sleep 10;
done

echo "Finishing...";
$SLURM_BIN/sacct
rm mywrapper.sh
