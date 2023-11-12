#!/bin/bash
DUNE_PATH="/Users/giacomosantoni/Desktop/TESI/Progetto_reco/mask-reco-tools"
TOOL_PATH="${DUNE_PATH}/tools-main"  
GEOMETRY_PATH="${DUNE_PATH}/geometry"


PRODDIR="${DUNE_PATH}/data/initial-data"   #reconstruction data folder
RECOFILE="${PRODDIR}/reco_data/3dreco-0-10.pkl"
PRIMARIESFILE="${PRODDIR}/numu_LBNF_GV2_prod_1.genie.edep-sim.root" 
SENSORSFILE="${PRODDIR}/sensors.root"
IDLIST="EDepInGrain_1.txt"    #text file with list of event numbers



python3 recodisplay.py ${RECOFILE} 265 ${GEOMETRY_PATH} -pe ${PRIMARIESFILE} -c 10 -m -ph ${SENSORSFILE} -it 300 -v 12 

###read event list from txt file and save reco pic for each event

# readarray -t -d='\n' EVENTLIST < ${IDLIST}
# echo ${EVENTLIST}
# touch ${PRODDIR}/savefig.log

# for EVN in ${EVENTLIST}
# do 
#   SAVEFILE="${PRODDIR}/reco_${EVN}.png" 
#   python3 recodisplay.py ${RECOFILE} ${EVN} ${GEOMETRY_PATH} -c 0.7 -eq -it 500 -v 12 -ph $SENSORSFILE -ssim -m -s $SAVEFILE 
# done
