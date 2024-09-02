#!/usr/bin/env bash
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0
(No options)
EOF
)

if [ $# -ne 0 ]; then
    log "Error: invalid command line arguments"
    log "${help_message}"
    exit 1
fi

curdir=$(dirname "$0")

nishika_datadir="${curdir}/../../../../../nishika-data"

datadir="${curdir}/../data"
pyscript="${curdir}/../../../../../src/format_data.py"


alldir="${datadir}/all"
traindir="${datadir}/train"
validdir="${datadir}/valid"
vrtestdir="${datadir}/valid_test"
testdir="${datadir}/test"

python $pyscript "${nishika_datadir}/train.csv" "${nishika_datadir}/train_details.csv" --output_dir "${datadir}/all"

./utils/subset_data_dir_tr_cv.sh ${alldir} ${traindir} ${vrtestdir}
./utils/subset_data_dir_tr_cv.sh ${vrtestdir} ${validdir} ${testdir}
