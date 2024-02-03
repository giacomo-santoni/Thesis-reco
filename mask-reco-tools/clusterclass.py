import numpy as np
from lpcclass import LPCSolver
from lpc_analysis_new import Clustering, Collinearity, defs, all_parameters

class VoxelCluster:
    def __init__(self, _centers, _amps):
        self.centers = _centers
        self.amps = _amps
        self.LPCs = []
        self.ComputeLPC(_centers, _amps)

    def ComputeLPC(self, centers, amps):
        lpc = LPCSolver()
        lpc.setPoints(centers, amps)
        lpc.setBandwidth(all_parameters["bandwidth"]) #find right parameter
        lpc.setStepSize(all_parameters["stepsize"]) #find right parameter  
        lpc.solve()
        self.LPCs.append(lpc.lpcPoints)

    def ExtendLPC(self, lpcPoints, geotree):
        R = 7*(geotree.voxel_size)
        externalPoints = []
        externalAmps = []
        for i,c in enumerate(self.centers):
            all_distances = []
            for lpc in lpcPoints:
                distance = np.linalg.norm(c - lpc)
                all_distances.append(distance)
            is_out = np.asarray(all_distances) > R
            if list(is_out).count(False) <= 2:#point is inside at most 2 spheres then it is external
                externalPoints.append(c)
                externalAmps.append(self.amps[i])

        if externalPoints:
            rem_clusters, _ = Clustering(externalPoints, externalAmps)
            for rem_clust in rem_clusters: 
                self.ComputeLPC(rem_clust.centers, rem_clust.amps)

    def ComputeLPCDistances(self):#compute vector distances only considering one lpc cluster
        all_distances = []#inizializzo qua perchè voglio le distanze tra punti per un singolo cluster
        for i in range(len(self.LPCs[0])):#loop sui singoli punti di un cluster di lpc
            if i < (len(self.LPCs[0])-1):
                distance = self.LPCs[0][i+1] - self.LPCs[0][i]
                all_distances.append(distance)
        return all_distances#in a single LPC cluster
    
    def FindBreakPoint(self):
        all_distances = self.ComputeLPCDistances()
        #t = all_parameters["stepsize"]
        #l = defs["voxel_size"]
        self.break_point = 0

        all_non_collinearities = []
        all_norms = []
        for i in range(len(all_distances)):
            if i < (len(all_distances)-1):
                scalar_product = np.dot(all_distances[i], all_distances[i+1])
                norm_product = np.abs(np.linalg.norm(all_distances[i]))*np.abs(np.linalg.norm(all_distances[i+1]))
                all_norms.append(norm_product)
                #given 2 vectors a and b, the non collinearity is |a||b|(1-|cosPhi|), that can be written also as |a||b| - a•b, since cosPhi = (a•b)/|a||b|
                cosine = Collinearity(all_distances[i], all_distances[i+1])
                non_collinearity =  (norm_product)*(1-np.abs(cosine))#(norm_product - np.abs(scalar_product))
                all_non_collinearities.append(non_collinearity)
        #make the non-coll values corresponding with the lpc points 
        #(starting from the second lpc point, where the first non-coll is computed)
        sorted_non_collinearities = {k:v for (k,v) in zip(range(1,len(self.LPCs[0])), all_non_collinearities)}
        for i,(k,v) in enumerate(sorted_non_collinearities.items()):
            if v/all_norms[i] > 0.094 and v == np.max(all_non_collinearities):#1-|cosPHI|>0.094 (i.e., PHI>20°) and it is the max value of non-collinearity
                self.break_point = k
    
    def BreakLPCs(self):#I have to pass the FIRST cluster of lpc points in a voxel cluster
        #_, break_point = self.ComputeLPCNonCollinearity()
        self.FindBreakPoint()
        self.broken_lpccurve = (self.LPCs[0][:self.break_point+1], self.LPCs[0][self.break_point:])

    # def MergeLPCPoints(self):
    #     closest_points = []
    #     min_distances = []
    #     for i in range(len(self.LPCs)):
    #         for j in range(i,len(self.LPCs)):
    #             distance1 = np.linalg.norm(self.LPCs[i][0] - self.LPCs[j][0])
    #             distance2 = np.linalg.norm(self.LPCs[i][0] - self.LPCs[j][-1])
    #             distance3 = np.linalg.norm(self.LPCs[i][-1] - self.LPCs[j][0])
    #             distance4 = np.linalg.norm(self.LPCs[i][-1] - self.LPCs[j][-1])
    #             index_diff = [[0,0], [0,-1], [-1,0], [-1,-1]]
    #             all_distances = [distance1, distance2, distance3, distance4]

    #             min_distance_index = np.argmin(all_distances)

    #             closest_points.append((i,j,index_diff[min_distance_index]))
    #             min_distances.append(np.min(all_distances))
    #         min_min_distance = np.argmin(min_distances)
    #         closest_points[min_min_distance]
            #condizione per capire se gli indici sono uguali o sono diversi: se sono uguali uno devo invertirlo, altrimenti se sono diversi va bene e posso concatenarli
                
 
