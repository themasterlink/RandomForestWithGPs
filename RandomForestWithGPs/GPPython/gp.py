#!/Users/Max/anaconda/bin/python
import numpy as np
import json
from pprint import pprint
import math
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
with open("init.json") as data_file:
    data = json.load(data_file)


class GaussianProccess:
    def __init__(self, fileName):
        lines = open(data["Training"]["path"], "r").read().split("\n")
        self.data = []
        self.lSquared = float(data["GP"]["l"]) * float(data["GP"]["l"])
        self.sigmaNSquared = float(data["GP"]["sigmaN"]) * float(data["GP"]["sigmaN"])
        self.labels = []
        for line in lines:
            if len(line) > 3:
                ele = line.split(",")
                point = np.array([float(ele[0]), float(ele[1])])
                self.data.append(point)
                self.labels.append(-1 if int(ele[2]) == 0 else 1)
        self.labels = np.asarray(self.labels)
        self.dataPoints = len(self.data)
        self.K = np.empty([self.dataPoints, self.dataPoints], dtype=float)
        for i in range(0, self.dataPoints):
            self.K[i][i] = self.sigmaNSquared
            for j in range(i + 1, self.dataPoints):
                temp = self.kernelOf(self.data[i], self.data[j])
                self.K[i][j] = temp
                self.K[j][i] = temp


    def updatePis(self):
        for i in range(0, self.dataPoints):
            self.pis[i] = 1.0 / (1.0 + math.exp(-self.labels[i] * self.f[i]))
            self.dPis[i] = self.t[i] - self.pis[i]
            self.ddPis[i] = -(-self.pis[i] * (1 - self.pis[i])) # - to get minus dd Pi
            self.sqrtDDPis[i] = math.sqrt(self.ddPis[i])

    def trainF(self):
        self.f = np.zeros(self.dataPoints)
        self.pis = np.empty(self.dataPoints)
        self.dPis = np.empty(self.dataPoints)
        self.ddPis = np.empty(self.dataPoints)
        self.sqrtDDPis = np.empty(self.dataPoints)
        self.t = (self.labels + np.ones(self.dataPoints)) * 0.5
        converge = False
        eye = np.eye(self.dataPoints)
        lastObject = 1e100;
        while(not converge):
            self.updatePis()
            self.W = np.diag(self.ddPis)
            self.WSqrt = np.diag(self.sqrtDDPis)
            C = eye + np.dot(np.dot(self.WSqrt, self.K), self.WSqrt)
            print("K:\n"+str(self.K))
            print("inner:\n"+str(C))
            self.L = scipy.linalg.cho_factor(C, lower = True)
            self.U = scipy.linalg.cho_factor(C, lower = False)
            b = np.dot(self.W, self.f) + self.dPis;
            nenner = scipy.linalg.cho_solve(self.L, (np.dot(self.WSqrt,np.dot(self.K,b))))
            self.a = b - np.dot(self.WSqrt, scipy.linalg.cho_solve(self.U, nenner))
            self.f = np.dot(self.K, self.a)
            prob = 1.0 / (1.0 + math.exp(-np.dot(self.labels,self.f)))
            objective = -0.5 * np.dot(self.f, self.a) + math.log(max(min(prob,1-1e-7),1e-7));
            print(objective)
            if math.fabs(objective / lastObject - 1.0) < 1e-5:
                converge = True
            lastObject = objective
        print("Trained")
        return

    def train(self):
        
        converge = False
        while(not converge):
            trainF()
            logZ = -0.5 * np.dot(self.a, self.f) + (-math.log(1 + math.exp(-np.dot(self.labels, self.f)))) + math.log(L.diagonal().sum())
            R = np.dot(self.WSqrt, scipy.linalg.cho_solve(self.U, scipy.linalg.cho_solve(self.L, self.WSqrt)))
            C = scipy.linalg.cho_solve(self.L, np.dot(self.WSqrt, self.K))
            dddPis = np.empty(self.dataPoints)
            for i in range(0, self.dataPoints):
                ddPis = -self.ddPis[i];
                dddPis[i] = - ddPis * (1-self.pis[i]) - self.pis[i] * (1 - ddPis)
            s2 = -0.5 * (self.K.diagonal() - np.dot(C.T,C).diagonal).diagonal() * dddPis
            #for i in range(0,3):
            #C =
            self.W = np.diag(self.ddPis)
            self.WSqrt = np.diag(self.sqrtDDPis)
            C = eye + np.dot(np.dot(self.WSqrt, self.K), self.WSqrt)
            self.L = scipy.linalg.cho_factor(C, lower = True)
            self.U = scipy.linalg.cho_factor(C, lower = False)
            b = np.dot(self.W, self.f) + self.dPis;
            nenner = scipy.linalg.cho_solve(self.L, (np.dot(self.WSqrt,np.dot(self.K,b))))
            a = b - np.dot(self.WSqrt, scipy.linalg.cho_solve(self.U, nenner))
            self.f = np.dot(self.K, a)
            prob = 1.0 / (1.0 + math.exp(-np.dot(self.labels,self.f)))
            objective = -0.5 * np.dot(self.f, a) + math.log(prob if prob > 1e-7 and prob < 1 - 1e-7 else 1e-7 if prob <= 1e-7 else 1 - 1e-7);
            print(objective)
                #if math.fabs(objective / lastObject - 1.0) < 1e-5:
            converge = True
            lastObject = objective
        print("Trained")
        return
    
    def predict(self, newPoint):
        kXStar = np.empty(self.dataPoints)
        for i in range(0, self.dataPoints):
            kXStar[i] = self.kernelOf(newPoint, self.data[i])
        fStar = np.dot(kXStar, self.dPis)
        v = scipy.linalg.cho_solve(self.L, np.dot(self.WSqrt,kXStar))
        vFStar = math.fabs(self.sigmaNSquared + 1 - np.dot(v,v))
        start = fStar - vFStar * 1.5
        end = fStar + vFStar * 1.5
        stepSize = (end - start) / float(data["GP"]["samplingAmount"])
        prob = 0.0
        for p in np.arange(start,end,stepSize):
            gaussRand = np.random.normal(fStar, vFStar)
            height = 1.0 / (1.0 + math.exp(p)) * gaussRand
            prob += height * stepSize;
        return max(min(prob,1), 0)

    def plot(self):
        plt.figure(0)
        min = np.min(self.data)
        max = np.max(self.data)
        min -= (max-min) * 0.2
        max += (max-min) * 0.2
        stepSize = (max - min) / float(data["GP"]["plotRes"]);
        listGrid = []
        i = 0
        for x in np.arange(min,max, stepSize):
            print("Done: " + str(float(i) / float(data["GP"]["plotRes"]) * 100) + "%")
            i += 1
            newList = []
            for y in np.arange(min,max, stepSize):
                newPoint = [y,x]
                prob = self.predict(newPoint)
                newList.append(prob)
            listGrid.append(newList)
        plt.imshow(listGrid, extent=(max, min, min, max), interpolation='nearest', cmap=cm.rainbow)
        plt.gca().invert_xaxis()
        plt.gca().set_ylim([min, max])
        plt.gca().set_xlim([min, max])
        for i in range(0,self.dataPoints):
            plt.plot(self.data[i][0],self.data[i][1], 'bo' if self.labels[i] == 1 else 'ro')
        print("Finished plotting")
        plt.show()

    def kernelOf(self, x, y):
        diff = x - y
        return math.exp(- 0.5 / self.lSquared * diff.dot(diff));

gp = GaussianProccess(data["Training"]["path"])
gp.trainF()
gp.plot()