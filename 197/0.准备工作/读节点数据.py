import pandas as pd
import numpy as np
data = pd.read_csv("附件2.csv",encoding="gbk")
a=data["对应主索节点编号"]
b=data['下端点X坐标（米）']
c=data['下端点Y坐标（米）']
d=data['下端点Z坐标（米）']
e=data['基准态时上端点X坐标（米）']
f=data['基准态时上端点Y坐标（米）']
g=data['基准态时上端点Z坐标（米）']
data_info节点={a[i]:([b[i],c[i],d[i]],[e[i],f[i],g[i]]) for i in range(2226)}