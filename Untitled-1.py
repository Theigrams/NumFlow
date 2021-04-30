# %%
from typing import List
class Node():
    def __init__(self, value: int = 0, parent: 'Node' = None) -> None:
        self.value = value
        self.parent = parent
        self.children: List['Node'] = []
        if parent:
            parent.children.append(self)
    def child_print(self):
        for child in self.children:
            print(child.value)
# %%
a = Node(value=1, parent=None)
b = Node(value=2, parent=a)

# %%
