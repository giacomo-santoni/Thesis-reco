import numpy as np
from lpcclass import LPCSolver
from lpc_analysis_new import Clustering, all_parameters, LPCDistances

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
        R = 6*(geotree.voxel_size)
        externalPoints = []
        externalAmps = []
        print(len(self.centers))
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

    def Angles(curve, all_LPCindex_points):
        all_clusters_all_distances = LPCDistances(all_curves)
        all_scalar_products = []
        all_LPC_vertex_nrpoints = []
        for i in range(len(all_clusters_all_distances)):#loop sui cluster di distanze (sono i cluster di lpc)
            scalar_products = []
            for j in range(len(all_clusters_all_distances[i])):#loop sulle singole distanze di un singolo cluster
                if j < (len(all_clusters_all_distances[i])-1):
                    scalar_prod = np.dot(all_clusters_all_distances[i][j], all_clusters_all_distances[i][j+1])
                    norm_product = np.linalg.norm(all_clusters_all_distances[i][j])*np.linalg.norm(all_clusters_all_distances[i][j+1])
                    quantity_to_plot = norm_product*(1 - np.abs(scalar_prod/norm_product))
            scalar_products.append(quantity_to_plot)
            lpc_point_angles = {k:v for (k,v) in zip(range(1,len(all_LPCindex_points[i])),scalar_products)}#faccio partire da 1 il conto perchè voglio che l'indice che conta gli angoli parta dal secondo punto lpc
            LPC_vertex_nrpoint = int([k for k, v in lpc_point_angles.items() if v == np.max(scalar_products)][0])
            all_LPC_vertex_nrpoints.append(LPC_vertex_nrpoint)
            all_scalar_products.append(scalar_products)
        return all_scalar_products, all_LPC_vertex_nrpoints



    def MergeLPCPoints(self):
        closest_points = []
        min_distances = []
        for i in range(len(self.LPCs)):
            for j in range(i,len(self.LPCs)):
                distance1 = np.linalg.norm(self.LPCs[i][0] - self.LPCs[j][0])
                distance2 = np.linalg.norm(self.LPCs[i][0] - self.LPCs[j][-1])
                distance3 = np.linalg.norm(self.LPCs[i][-1] - self.LPCs[j][0])
                distance4 = np.linalg.norm(self.LPCs[i][-1] - self.LPCs[j][-1])
                index_diff = [[0,0], [0,-1], [-1,0], [-1,-1]]
                all_distances = [distance1, distance2, distance3, distance4]

                min_distance_index = np.argmin(all_distances)

                closest_points.append((i,j,index_diff[min_distance_index]))
                min_distances.append(np.min(all_distances))
            min_min_distance = np.argmin(min_distances)
            closest_points[min_min_distance]
            #condizione per capire se gli indici sono uguali o sono diversi: se sono uguali uno devo invertirlo, altrimenti se sono diversi va bene e posso concatenarli
                
 
