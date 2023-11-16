# Thesis-reco
GOAL: apply the LPC algorithm on several patterns of event, in particular events with 2 and 3 tracks. Hence, I will take the lpc algorithm already implemented and I will improve it generalizing its usage for other patterns of events. I will have to be careful also to the presence of clusters. Their presence depends on the cut we choose. To avoid a too high influence of the cut in the analysis, we try to perform an analysis through the gradient: so we plot the gradient of the voxels instead of the voxel amplitude itself. In this way, the event is quite clear after half the number of iterations with respect to the other analysis.
An important part of this analysis is the choice of the parameters. The parameters we have to choose are: 
for DBSCAN algorithm that identifies clusters: 
* epsilon and minPoints: epsilon chosen looking at the kneighbour distance; minPoints as 2*nr_dim
for LPC:
* stepsize and bandwidth: chosen similar and similar to epsilon(?)
and the cut on the voxel amplitude