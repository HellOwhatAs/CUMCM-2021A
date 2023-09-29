import shelve
from numba import njit
from math import pi,sin,cos,radians
import numpy as np

with shelve.open("this_dataset") as dat:
    data节点=dat["data节点"]
    data_info节点=dat["data_info节点"]
    data面板=dat["data面板"]

@njit#向量的模
def pjfunc(x:np.ndarray):
    return np.sqrt(np.sum(np.square(x)))
R=0
for i in data节点:
    R+=pjfunc(np.array(data节点[i]))
R/=len(data节点)

@njit#单位化
def dwh(x:np.ndarray):
    return x/np.sqrt(np.sum(np.square(x)))
@njit#两点间距离
def distance(x:np.ndarray,y:np.ndarray):
    return np.sqrt(np.sum(np.square(x-y)))
@njit
def _法向量(a:np.ndarray,b:np.ndarray,c:np.ndarray):
    return np.cross(a-b,a-c)
def roll(x:np.ndarray,rol,pit,yaw):
    """x,y,z"""
    Rx=np.array([
        [1,0,0],
        [0,cos(rol),-sin(rol)],
        [0,sin(rol),cos(rol)]
    ])
    Ry=np.array([
        [cos(pit),0,sin(pit)],
        [0,1,0],
        [-sin(pit),0,cos(pit)]
    ])
    Rz=np.array([
        [cos(yaw),-sin(yaw),0],
        [sin(yaw),cos(yaw),0],
        [0,0,1]
    ])
    return Rz@Ry@Rx@x
def func(x:np.ndarray):
    zz=roll(x,0,0,-36.795*pi/180)
    zz=roll(zz,0,-radians(90-78.169),0)
    for k in range(3):
        x[k]=zz[k]
def rfunc(x:np.ndarray):
    zz=roll(x,0,radians(90-78.169),0)
    zz=roll(zz,0,0,36.795*pi/180)
    for k in range(3):
        x[k]=zz[k]
# 变换后的坐标系里面的抛物面方程：z=0.001781612254082338*(x^2+y^2)-300.735945677618
@njit#抛物面
def pwm(x,y):
    return 0.001781612254082338*(x**2+y**2)-300.735945677618
aa=np.array([0,0,-300.735945677618])
@njit#抛物面的法向量
def fpwm(x,y):
    return np.array([-2*0.001781612254082338*x,-2*0.001781612254082338*y,1])
@njit#d是否在abc三角形中
def in_tr(a:np.ndarray,b:np.ndarray,c:np.ndarray,d:np.ndarray):
    fa=lambda x,y:((x-b[0])*(b[1]-c[1])-(y-b[1])*(b[0]-c[0]))
    fb=lambda x,y:((x-a[0])*(a[1]-c[1])-(y-a[1])*(a[0]-c[0]))
    fc=lambda x,y:((x-b[0])*(b[1]-a[1])-(y-b[1])*(b[0]-a[0]))
    return fa(d[0],d[1])*fa(a[0],a[1])>0 and fb(d[0],d[1])*fb(b[0],b[1])>0 and fc(d[0],d[1])*fc(c[0],c[1])>0
@njit#_面板.train得到平均法向量
def _train(a:np.ndarray,b:np.ndarray,c:np.ndarray):
    x0=min(a[0],b[0],c[0])
    x1=max(a[0],b[0],c[0])
    y0=min(a[1],b[1],c[1])
    y1=max(a[1],b[1],c[1])
    x=np.linspace(x0,x1)
    y=np.linspace(y0,y1)
    fxl=np.zeros((3,))
    total_n=0
    for i in x:
        for j in y:
            if in_tr(a,b,c,np.array([i,j])):
                fxl+=fpwm(i,j)
                total_n+=1
    return fxl/total_n
@njit#球坐标下点到抛物面的距离
def distance_to_pwm(pos:np.ndarray):
    x,y,z=pos[0],pos[1],pos[2]
    a,b=0.001781612254082338,300.735945677618
    return ((x**2+y**2+z**2)*(2*a*(x**2+y**2)-z-(4*a*b*(x**2+y**2)+z**2)**0.5)**2)**0.5/(2*a*(x**2+y**2))
@njit
def sqrt(x):
    return x**0.5
@njit#点到抛物面的最短距离
def min_distance_topwm(pos:np.ndarray):
    a,b=0.001781612254082338,300.735945677618
    x,y,z=pos[0],pos[1],pos[2]
    return sqrt(((sqrt(3)*a**2*sqrt((27*a**2*(x**2 + y**2) - 2*(2*a*b + 2*a*z - 1)**3)/a**6) + 9*sqrt(x**2 + y**2))/a**2)**(2/3)*(36*a**2*((sqrt(3)*a**2*sqrt((27*a**2*(x**2 + y**2) - 2*(2*a*b + 2*a*z - 1)**3)/a**6) + 9*sqrt(x**2 + y**2))/a**2)**(2/3)*(6*a**2*((sqrt(3)*a**2*sqrt((27*a**2*(x**2 + y**2) - 2*(2*a*b + 2*a*z - 1)**3)/a**6) + 9*sqrt(x**2 + y**2))/a**2)**(1/3)*sqrt(x**2 + y**2) - 6**(1/3)*(a**2*((sqrt(3)*a**2*sqrt((27*a**2*(x**2 + y**2) - 2*(2*a*b + 2*a*z - 1)**3)/a**6) + 9*sqrt(x**2 + y**2))/a**2)**(2/3) + 6**(1/3)*(2*a*b + 2*a*z - 1)))**2 + (36*a**3*((sqrt(3)*a**2*sqrt((27*a**2*(x**2 + y**2) - 2*(2*a*b + 2*a*z - 1)**3)/a**6) + 9*sqrt(x**2 + y**2))/a**2)**(2/3)*(b + z) - 6**(2/3)*(a**2*((sqrt(3)*a**2*sqrt((27*a**2*(x**2 + y**2) - 2*(2*a*b + 2*a*z - 1)**3)/a**6) + 9*sqrt(x**2 + y**2))/a**2)**(2/3) - 6**(1/3)*(-2*a*b - 2*a*z + 1))**2)**2)/(a**2*(sqrt(3)*a**2*sqrt((27*a**2*(x**2 + y**2) - 2*(2*a*b + 2*a*z - 1)**3)/a**6) + 9*sqrt(x**2 + y**2))**2))/36
@njit#_节点.move_to_pwm
def _dpos_to_pwm(pos:np.ndarray):
    dwpos=pos/np.sum(pos)
    distan=1000000
    keep_i=0
    for i in np.linspace(-0.6,0.6,10000):#pos+dwpos*i
        # b=(pos+dwpos*i)
        # a=abs(b[2]-pwm(b[0],b[1]))
        a=distance_to_pwm(pos+dwpos*i)
        if a<distan:
            distan=a
            keep_i=i
    return keep_i*dwpos,distan
@njit#获取abc组成的面板在a那边的1/3散点
def _getallpos(a:np.ndarray,b:np.ndarray,c:np.ndarray):
    AB=b-a
    AC=c-a
    AB/=8
    AC/=8
    n=0
    reta=np.zeros((17,3),dtype=np.float64)
    for i in range(5):
        for j in range(5-i):
            reta[n]+=AC*i+AB*j
            n+=1
    reta[n]+=AC*3+AB*2
    n+=1
    reta[n]+=AC*2+AB*3
    return reta+a
@njit#将上一个函数的返回值变成在球面上的点
def getallpos2ball(x:np.ndarray,center:np.ndarray):
    for i in range(len(x)):
        x[i]=x[i]-center
        x[i]=x[i]*R/np.sqrt(np.sum(np.square(x[i])))
        x[i]=x[i]+center
    return x
@njit#计算1/3块板上的散点到抛物面的最短距离的平均值
def _mean_distance2pwm(ii:np.ndarray):
    num=0
    total_distance=0
    for i in range(17):
        a=min_distance_topwm(ii[i])
        if i==0:
            f=1/6
        elif i==1 or i==2 or i==3 or i==4 or i==5 or i==9 or i==12 or i==14:
            f=1/2
        else:
            f=1.
        if ii[i][2]>pwm(ii[i][0],ii[i][1]):
            t=-1
        else:
            t=1
        num+=f
        total_distance+=f*a*t
    return total_distance,num
@njit
def diancheng(a:np.ndarray,b:np.ndarray):
    ret=0
    for i in range(a.shape[0]):
        ret+=a[i]*b[i]
    return ret
@njit
def _duichen(ruse:np.ndarray,fxl:np.ndarray):
    b=fxl/np.sqrt(np.sum(np.square(fxl)))
    b*=diancheng(b,ruse)
    c=(b-ruse)*2
    return ruse+c
@njit
def _refdot(pos1:np.ndarray,fxxl:np.ndarray,z=-160.2):
    if fxxl[2]<0:
        fxxl*=-1
    f1=(z-pos1[2])/fxxl[2]
    p1=pos1+f1*fxxl
    if p1[0]**2+p1[1]**2<0.25:
        return 1.
    else:
        return 0.
    # return p1
@njit
def _法向量(a:np.ndarray,b:np.ndarray,c:np.ndarray):
    return np.cross(a-b,a-c)
@njit
def _refdots(dj1:np.ndarray,dj2:np.ndarray,dj3:np.ndarray,center:np.ndarray):
    all_lights=0
    geted_lights=0
    ret=np.zeros((17*3,3))
    for i in range(17):
        if i==0:
            f=1/6
        elif i==1 or i==2 or i==3 or i==5 or i==9 or i==12 or i==15 or i==16:
            f=0.5
        elif i==4 or i==14:
            f=0.25
        else:
            f=1.
        # ret[i]+=(_refdot(dj1[i],_duichen(np.array([0,0,1]),center-dj1[i])))
        # ret[17+i]+=(_refdot(dj2[i],_duichen(np.array([0,0,1]),center-dj2[i])))
        # ret[17*2+i]+=(_refdot(dj3[i],_duichen(np.array([0,0,1]),center-dj3[i])))
        all_lights+=3*f
        geted_lights+=(_refdot(dj1[i],_duichen(np.array([0,0,1]),center)))*f
        geted_lights+=(_refdot(dj2[i],_duichen(np.array([0,0,1]),center)))*f
        geted_lights+=(_refdot(dj3[i],_duichen(np.array([0,0,1]),center)))*f
    return geted_lights,all_lights
    # return ret
@njit# 计算三角形面积
def tr_square(a:np.ndarray,b:np.ndarray,c:np.ndarray):
    return abs(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))/2

class _节点:
    def __init__(self,name,pos,info_pos):
        self.name=name
        self.pos=np.array(pos)
        self.xpos,self.spos=np.array(info_pos[0]),np.array(info_pos[1])
        self.board=[]
        self.boarddots=dict()
    def move(self,x:float):# 移动节点（向上为正
        fxxl=self.spos-self.xpos
        fxxl/=pjfunc(fxxl)
        if fxxl[2]<0:
            fxxl*=-1
        dpos=fxxl*x
        self.pos+=dpos
        self.spos+=dpos
        self.xpos+=dpos
    def move_to_pwm(self):# 将自己贴到抛物面上（在伸缩范围内）
        dpos,distan=_dpos_to_pwm(self.pos)
        self.pos+=dpos
        self.spos+=dpos
        self.xpos+=dpos
        return distan
    def mean_distance2pwm(self):# 六边形的散点到抛物面的平均距离（自己在抛物面的下边为正）
        mnum=0
        mtotal_distance=0
        for i in self.boarddots:
            au,bu=_mean_distance2pwm(self.boarddots[i])
            mnum+=bu
            mtotal_distance+=au
        return mtotal_distance/mnum
    def __str__(self):
        return "<节点"+self.name+str(self.pos)+'>'
    def __repr__(self):
        return "<节点"+self.name+str(self.pos)+"下"+str(self.xpos)+"上"+str(self.spos)+'>'

d节点={i:_节点(i,data节点[i],data_info节点[i]) for i in data节点}# 所有的节点
# dr节点={i:_节点(i,data节点[i],data_info节点[i]) for i in data节点}
class _面板:
    def __init__(self,d1:_节点,d2:_节点,d3:_节点,ind:int):
        self.ds=[d1,d2,d3]
        self.bc=np.array([distance(d1.pos,d2.pos),distance(d2.pos,d3.pos),distance(d3.pos,d1.pos)])# 初始三边长度
        self.ind=ind#索引
        d1.board.append(ind)
        d2.board.append(ind)
        d3.board.append(ind)
        self.expected_fxl=None
    def train(self):
        self.expected_fxl=_train(*(self.ds[i].pos for i in range(3)))
    def getallpos(self):# 将散点给到自己的三个顶点
        centerofball=_法向量(*(self.ds[i].pos for i in range(3)))#root(lambda x:np.array([distance(self.ds[0].pos,x)-R,distance(self.ds[1].pos,x)-R,distance(self.ds[2].pos,x)-R]),[0,0,0]).x
        self.ds[0].boarddots[self.ind]=((_getallpos(self.ds[0].pos,self.ds[1].pos,self.ds[2].pos)))
        self.ds[1].boarddots[self.ind]=((_getallpos(self.ds[1].pos,self.ds[2].pos,self.ds[0].pos)))
        self.ds[2].boarddots[self.ind]=((_getallpos(self.ds[2].pos,self.ds[0].pos,self.ds[1].pos)))
    def refarea(self):
        centerofball=_法向量(*(self.ds[i].pos for i in range(3)))#root(lambda x:np.array([distance(self.ds[0].pos,x)-R,distance(self.ds[1].pos,x)-R,distance(self.ds[2].pos,x)-R]),[0,0,0]).x
        percent = _refdots(
            (_getallpos(self.ds[0].pos,self.ds[1].pos,self.ds[2].pos)),
            (_getallpos(self.ds[1].pos,self.ds[2].pos,self.ds[0].pos)),
            (_getallpos(self.ds[2].pos,self.ds[0].pos,self.ds[1].pos)),
            centerofball
        )
        return percent[0]/percent[1],tr_square(self.ds[0].pos,self.ds[1].pos,self.ds[2].pos)
        
    def __str__(self):
        return "<面板:{}:{}:{}>".format(*(i.name for i in self.ds))
    def __repr__(self):
        return "<面板:{}:{}:{}>".format(*(i for i in self.ds))
d面板=[_面板(*(d节点[j] for j in i),ii) for ii,i in enumerate(data面板)]# 所有的面板

for i in d节点:# 坐标变换
    func(d节点[i].pos)
    func(d节点[i].spos)
    func(d节点[i].xpos)

in_r150节点=set()# 在新坐标系下位于口径300内的节点
@njit#平面上的点到原点的距离
def _f(a,b):
    return (a**2+b**2)**0.5
for i in d节点:
    if _f(d节点[i].pos[0],d节点[i].pos[1])<150:
        in_r150节点.add(i)# 统计150以内的点

@njit# 计算带有权重的17个点的平方和
def pingfang17(x:np.ndarray):
    ret=0.
    for i in range(17):
        if i==0:
            f=1/6
        elif i==1 or i==2 or i==3 or i==4 or i==5 or i==9 or i==12 or i==14 or i==15 or i==16:
            f=1/2
        else:
            f=1.
        ret+=min_distance_topwm(x[i])**2*f
    return ret

for i in d面板:
    i.getallpos()# 更新散点
pre=-100
while True:# 调整位置
    fang=0
    total=0# * 12+1/6
    for i in in_r150节点:
        d=d节点[i].mean_distance2pwm()
        d节点[i].move(d)
        for ooioo in d节点[i].board:
            d面板[ooioo].getallpos()
        fang+=d**2
        total+=1
    h=sqrt(fang/total)#*(12+1/6)
    print("均方根误差:",h)
    if abs(pre-h)<0.0001:
        break
    pre=h

totao_perc,all_area=0,0
for i in d面板:
    if i.ds[0].name in in_r150节点 and i.ds[1].name in in_r150节点 and i.ds[2].name in in_r150节点: 
        percen,area=i.refarea()
        totao_perc+=percen*area
        all_area+=area
print("反射面板是平面的工作抛物面接收率:",totao_perc/all_area)
