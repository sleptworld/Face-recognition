"""
Let's do something!
"""
import classFunc
import matplotlib.pyplot as plt
from PIL import Image

print(__doc__)

path = './data/nowear/'

t = classFunc.test(path,100,(216,176))

trainData = t.getTrainData()

testData = t.getTestData()

p = t.down(29,trainData)

print(p)