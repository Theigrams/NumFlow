from Demo_Node import Node
class MatMul(Node):
    """
    矩阵乘法
    """
    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape()[1] == self.parents[1].shape()[0]
        self.value=self.parents[0].value @ self.parents[1].value
    
    def get_jacobi(self, parent: Node):
        """
        计算其关于某个父节点的雅可比矩阵
        """
        