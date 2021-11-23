from model.utils import *
from model.DL_ClassifierModel import *

dataClass = DataClass('data.txt', validSize=0.2, testSize=0.0, kmers=3)
dataClass.vectorize("char2vec", feaSize=64)
s, f, k, d = 64, 128, 3, 64
model = TextClassifier_SPPCNN(classNum=5, embedding=dataClass.vector['embedding'], SPPSize=s, feaSize=d, filterNum=f, contextSizeList=[1, 3, 5], embDropout=0.3, fcDropout=0.5, useFocalLoss=True, device="cuda")
model.cv_train(dataClass, trainSize=1, batchSize=16, stopRounds=200, earlyStop=10, epoch=100, kFold=5, savePath=f"out/DeepLncLoc_s{s}_f{f}_k{k}_d{d}", report=['ACC', 'MaF', 'MiAUC', 'MaAUC'])