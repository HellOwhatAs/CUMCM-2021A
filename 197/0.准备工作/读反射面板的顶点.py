import pandas as pd
data = pd.read_csv("附件3.csv",encoding="gbk")
a=data['主索节点1']
b=data['主索节点2']
c=data['主索节点3']
data面板=[(a[i],b[i],c[i]) for i in range(4300)]
# print(data面板)