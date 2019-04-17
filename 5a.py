from sklearn import metrics
from collections import Counter
from scipy.spatial.distance import pdist,squareform
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from apyori import apriori
from tqdm import tqdm

store_data = pd.read_csv('patterns_BlackFriday.csv')  

records = []  
for i in range(199):  
    records.append([str(store_data.values[i,j]) for j in range(8)])

association_rules = apriori(records, min_support=0.029, min_lift=2, min_length=3)  
association_results = list(association_rules) 


for item in association_results:

    pair = item[0] 
    items = [x for x in pair]
    confidence =  item[2][0][2]
    lift =  item[2][0][3]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    support_AC = item[1]
    support_A = support_AC / confidence
    support_C = confidence / lift
    leverage = support_AC - support_A*support_C

    #print("Support: " , support_AC)
    #print("Confidence: " , confidence)
    #print("Lift: " ,lift)
    #print("Leverage: " , leverage)
    print("=====================================")