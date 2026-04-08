import enum
import matplotlib.pyplot as plt

class Status(enum.Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3

class Node:
    def __init__(self, name):
        self.name = name

    def tick(self, blackboard):
        raise NotImplementedError

    def get_tree_string(self, indent=''):
        return indent + f"- {self.name} ({self.__class__.__name__})\n"

class Composite(Node):
    def __init__(self, name, children=None):
        super().__init__(name)
        self.children = children if children is not None else []

    def get_tree_string(self, indent=''):
        s = super().get_tree_string(indent)
        for child in self.children:
            s += child.get_tree_string(indent + '  ')
        return s

class Selector(Composite):
    """
    Executes children sequentially until one succeeds.
    """
    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status != Status.FAILURE:
                return status
        return Status.FAILURE

class Sequence(Composite):
    """
    Executes children sequentially until one fails.
    """
    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS

class Action(Node):
    def __init__(self, name):
        super().__init__(name)

    def tick(self, blackboard):
        # This is where the action logic would go.
        # For this example, we'll just print the action name.
        print(f"Action: {self.name}")
        return Status.SUCCESS

class Blackboard:
    def __init__(self):
        self._data = {}

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value

def plot_behavior_tree(root_node, filename):
    """
    Generates a text-based plot of the behavior tree and saves it to a file.
    """
    tree_string = root_node.get_tree_string()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.05, 0.95, tree_string, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace')
    ax.axis('off')
    plt.title("Behavior Tree Structure")
    plt.savefig(filename)
    plt.close()
