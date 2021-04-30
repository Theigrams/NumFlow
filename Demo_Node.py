#%%
from typing import List
import numpy as np
from .Demo_Matrix import Mat
from .Demo_Graph import Graph, default_graph
#%%
class Node():
    """
    c = a + b
    那么 c 就是 a 和 b 的子节点
    """

    def __init__(self, *parents, **kargs) -> None:
        # 定义计算图
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        self.gen_node_name(**kargs)

        self.parents: List['Node'] = list(parents)
        self.children: List['Node'] = []
        self.value : Mat = None
        self.jacobi : Mat = None

        # 将该节点插入到其父节点的子节点列表中
        for parent in self.parents:
            parent.children.append(self)

        # 将该节点加入到计算图中
        self.graph.add_node(self)

    def get_parents(self):
        return self.parents

    def get_children(self):
        return self.children

    def gen_node_name(self, **kargs):
        """
        生成节点名称
        若用户指定了名称，则在前面加上 name_scope
        """
        self.name = kargs.get('name', '{}:{}'.format(
            self.__class__.__name__, self.graph.node_count()))
        if self.graph.name_scope:
            self.name = '{}/{}'.format(self.graph.name_scope, self.name)

    def forward(self):
        """
        Foward 操作是计算该点处的值
        先对父节点递归调用 foward, 计算出父节点的值
        然后统一求和
        """
        for node in self.parents:
            if node.value is None:
                node.forward()

        self.compute()

    @abc.abstractmethod
    def compute(self):
        """
        抽象方法，需根据节点情况定义
        举例: node 处的值为其父节点的值之和
        self.value = np.zeros(self.parents[0].shape())
        for parent in self.parents:
            self.value += parent.value
        """
        pass

    def backward(self, result: 'Node'):
        """
        result 为一个向量
        返回 node 关于 result 的 jacobi 矩阵
        c = 2a + 3b
        d = a**2
        e = c + 2d
        a.foward(e):
            a.jcb = c.jcb * c.get_jcb(a) + d.jcb * d.get_jcb(a) 
        """
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.eye(self.dimension())
            else:
                self.jacobi = np.zeros((result.dimension(), self.dimension()))
                for child in self.get_children():
                    if child.value is not None:
                        self.jacobi += child.backward(result) * \
                            child.get_jacobi(self)
        return self.jacobi

    @abc.abstractmethod
    def get_jacobi(self, parent: 'Node'):
        """
        抽象方法
        举例:
        c = a**2
        c.get_jbc(a):
            return 2*np.diag(parent.value)
        """
        pass

    def clear_jacobi(self):
        self.jacobi = None

    def dimension(self):
        """
        该节点展开后的维数
        """
        return self.value.shape[0]*self.value.shape[1]

    def shape(self):
        return self.value.shape

    def reset_value(self, recursive: bool = True):
        """
        重置该节点处的值
        """
        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()

#%%
class Variable(Node):
    """
    变量节点
    暂时不知道用来干嘛
    """
    def __init__(self, dim, init=False,trainalbe=True , **kargs) -> None:
        super().__init__( **kargs)
        self.dim = dim

        # 如果数据需要初始化，那么初始化为高斯分布
        if init:
            self.value=np.random.normal(0,0.001,self.dim)
        
        # 设置变量是否参与训练
        self.trainable = trainalbe
    
    def set_value(self,value):
        assert isinstance(value, Mat) and value.shape == self.dim

        # 若本节点的值被修改，则也修改下游节点的值
        # 感觉岂不是会重复修改很多次
        # 也不一定，可能会在 node.compute 里重新计算
        self.reset_value()
        self.value = value
        

