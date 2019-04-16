import pyfpgrowth
import pandas as pd
store_data = pd.read_csv('small_BlackFriday.csv') 
records = []  
for i in range(148):  
    records.append([str(store_data.values[i,j]) for j in range(3)])

'''
transactions = [[1, 2, 5],
                [2, 4],
                [2, 3],
                [1, 2, 4],
                [1, 3],
                [2, 3],
                [1, 3],
                [1, 2, 3, 5],
                [1, 2, 3]]
'''

patterns = pyfpgrowth.find_frequent_patterns(records, 0.1)
rules = pyfpgrowth.generate_association_rules(patterns, 1)

print "=============Patterns============"
print patterns 
print "=============Rules============"
print rules