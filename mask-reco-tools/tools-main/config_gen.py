#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:47:59 2021

@author: alessandro
"""
#%% importing the necessary libraries
import pandas as pd 
import argparse as ap
import os
#%% Main function code
if  __name__ == "__main__":
    parser = ap.ArgumentParser(description="Reading of an excel file")
    parser.add_argument('--path',help="select the file path",default='./')
    parser.add_argument('--name',help="select the file name",default='Configs.csv')
    parser.add_argument('--folder',help='select the destination folder',default='./configs')
    args=parser.parse_args()
    try:
        os.mkdir(args.folder)
    except FileExistsError:
        pass
    df = pd.read_csv('{0}{1}'.format(args.path,args.name),delimiter='\s+')
    #df=pd.DataFrame(data,columns=['mask_type','mask_rank','b','dim','hole_pitch','mask_hole','det_side','det_cellcount','side_width','distance'])
    print(df)
    template='''folder = ../geometries/MASK/config{0}
mask_type = {1}
mask_rank = {2}
b = {3}
dim = {4}
hole_pitch = {5}
mask_hole = {6}
det_side = {7}
det_cellcount = {8}
side_width = {9}
distance = {10}
    '''
    for cf in df.itertuples():
        with open('{0}/config{1}'.format(args.folder,cf[0]), 'w') as s:
            print(template.format(cf[0],cf[1],cf[2],cf[3],cf[4],cf[5],cf[6],cf[7],cf[8],cf[9],cf[10]),file=s)