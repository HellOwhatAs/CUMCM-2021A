from 读结点 import data节点
from 读节点数据 import data_info节点
from 读反射面板的顶点 import data面板
import shelve
with shelve.open("this_dataset") as d:
    d["data节点"]=data节点
    d["data_info节点"]=data_info节点
    d["data面板"]=data面板
