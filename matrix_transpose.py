from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession
from numpy import genfromtxt


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: matrix_symmetric <input file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("Python_matrix_symmetric")\
        .getOrCreate()
                
    def f(x):
        if x[1]>x[0]:
            x[0],x[1] = x[1],x[0]
        return tuple(x)                        

    def y(a,b):
        if a[0]==b[0]:
            a[1]+=1
            b[1]+=1
        return(a,b)    
    
    def z(a):
        if a[1]!=2:
            return(1)
        else:
            return(0) 

    lines = spark.read.text(sys.argv[1]).rdd\
                .map(lambda r: r[0].split(','))\
                .map(lambda x: [int(y) for y in x])\
                .filter(lambda x: x[0]!=x[1])\
                .map(f)\
                .map(lambda x: (x,1))\
                .reduceByKey(add)

    count = lines.map(z).reduce(add)

    print(not count)

    spark.stop()
