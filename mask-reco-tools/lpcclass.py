import numpy as np
from numpy.linalg import eig
import scipy.linalg as la
import matplotlib.pyplot as plt 


class LPCSolver:
    def __init__(self) -> None:
        self.lpcPoints = []
        self.maxPoints = 200
        self.corvergenceRate = 10e-8
        self.correctionFactor = 1
        self.isDirectionForwards = True
        self.h = 80   #constant bandwidth parameter
        self.stepSize = 40
    
    def setMaxPoints(self, _maxPoints):
        self.maxPoints = _maxPoints

    def setBandwidth(self, _h):
        self.h = _h
    
    def setStepSize(self, _stepSize):
        self.stepSize= _stepSize
    
    def setPoints(self, _points, _amps):
        self.pointsXYZ = _points
        self.pointsAmpl = _amps

    def setMaxAmpPoint(self, _points, _amps):
        pointswithAmps = dict(zip(_amps, _points))
        maxAmpPoint = 0
        for amp in pointswithAmps.keys():
            if amp == np.max(list(pointswithAmps.keys())):
                maxAmpPoint = pointswithAmps[amp]
        return maxAmpPoint

    def __computeWeight(self, location, pointAmpl, point) -> float:
        w = pointAmpl / (np.sqrt((2 * np.pi) ** 3) * (self.h ** 3))
        w *= np.exp(-1 /( 2* self.h * self.h ) * np.dot(point - location, point - location))
        return w 
    
    def __meanShift(self, location):#è la LOCAL MEAN
        vecSum = np.zeros(3)
        weightSum = 0
        for i,p in enumerate(self.pointsXYZ):
            w = self.__computeWeight(location, self.pointsAmpl[i], p)
            vecSum += w * p
            weightSum += w 
        return vecSum/weightSum

    def __lpcShift(self, location, prevEigVec):
        vecSum = np.zeros(shape=(3,3))
        weightSum = 0
        for i,p in enumerate(self.pointsXYZ):
            w = self.__computeWeight(location, self.pointsAmpl[i], p)
            vecSum += w * np.outer(p - location, p - location)#prodotto esterno
            weightSum += w
        covMatrix = vecSum / weightSum
        eigVal, eigVec = np.linalg.eig(covMatrix)
        lpcShiftEigVec = eigVec[np.argmax(eigVal)] / la.norm(eigVec[np.argmax(eigVal)])
        anglePenalization = 1
        if prevEigVec is not None:
            cosPhi = np.dot(lpcShiftEigVec, prevEigVec) / (la.norm(prevEigVec) * la.norm(lpcShiftEigVec))
            anglePenalization = np.abs(cosPhi) ** 2
            if (cosPhi < 0):
                lpcShiftEigVec = - lpcShiftEigVec
        if self.isDirectionForwards:
            lpcShiftPoint = location + lpcShiftEigVec * self.stepSize * anglePenalization
        else :
            lpcShiftPoint = location - lpcShiftEigVec * self.stepSize * anglePenalization
        return lpcShiftPoint, lpcShiftEigVec

    def __checkConvergence(self, thisPathLength, totalPathLength):
        isConverged = False
        R = thisPathLength / totalPathLength
        if R < self.corvergenceRate:
            isConverged = True
        return isConverged


    def solve(self):
        #find start point as points centroid (amplitude-weighted )
        self.isDirectionForwards = True 
        self.startPoint = np.average(self.pointsXYZ, axis = 0, weights = self.pointsAmpl)
        #self.startPoint = self.setMaxAmpPoint(self.pointsXYZ, self.pointsAmpl)
        print(self.startPoint)
        previousPoint = self.__meanShift(self.startPoint)
        self.lpcPoints.append(previousPoint)     
        previousEigVec = None
        nPoints = 1
        totalPathLength = 0
        #forward direction
        while (nPoints < self.maxPoints/2):
            nextPoint, nextEigVec = self.__lpcShift(previousPoint, previousEigVec)
            nextPoint = self.__meanShift(nextPoint)
            thisPathLength = la.norm(nextPoint - previousPoint)
            totalPathLength += thisPathLength
            self.lpcPoints.append(nextPoint)
            previousPoint = nextPoint
            previousEigVec = nextEigVec
            nPoints += 1
            if (self.__checkConvergence(thisPathLength, totalPathLength)):
                break
            
        # back first lpc point (local mean in start point), now go backwards
        self.isDirectionForwards = False
        previousEigVec = None     
        previousPoint = self.lpcPoints[0]
        totalPathLength = 0
        while (nPoints < self.maxPoints):
            nextPoint, nextEigVec = self.__lpcShift(previousPoint, previousEigVec)
            nextPoint = self.__meanShift(nextPoint)
            thisPathLength = la.norm(nextPoint - previousPoint)
            totalPathLength += thisPathLength
            self.lpcPoints.append(nextPoint)
            previousPoint = nextPoint
            previousEigVec = nextEigVec
            nPoints += 1 
            if (self.__checkConvergence(thisPathLength, totalPathLength)):
                break
            
        self.lpcPoints = np.array(self.lpcPoints)

    def plot(self, truth=None):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.lpcPoints[:, 0], self.lpcPoints[:,1], self.lpcPoints[:,2], color = 'red')
        ax.scatter3D(self.pointsXYZ[:, 0], self.pointsXYZ[:,1], self.pointsXYZ[:,2], c = self.pointsAmpl, cmap = 'viridis', alpha = 0.3)
        ax.scatter3D(self.lpcPoints[0, 0], self.lpcPoints[0,1], self.lpcPoints[0,2], color = 'green')
        if truth is not None:
            dirh = truth[0]
            dirp = truth[1]
            ax.quiver(*dirh, *dirp*10, color='xkcd:salmon', arrow_length_ratio=0.1, linewidth=2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        #plt.show()
