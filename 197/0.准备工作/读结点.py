import pandas as pd
import numpy as np
data = pd.read_csv("附件1.csv",encoding="gbk")
a=data['节点编号']
b=data['X坐标（米）']
c=data['Y坐标（米）']
d=data['Z坐标（米）']
data节点={a[i]:[b[i],c[i],d[i]] for i in range(2226)}