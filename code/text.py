# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def a(self, root, ll):
        if root is None:
            return
        list = []
        self.xian(root, 0, list)
        ll.extend(list)
        self.a()

    def xian(self, root, sum, list):
        if root is None:
            return
        sum += root.val
        list.append(sum)
        self.xian(root.left, sum, list)
        self.xian(root.right, sum, list)


if __name__ == '__main__':
    root = TreeNode(1)
    root.left = TreeNode(3)
    root.right = TreeNode(1)
    p = root.left
    p.left = TreeNode(4)
    p.right = TreeNode(5)

    q = root.right
    q.left = TreeNode(6)
    q.right = TreeNode(7)

    s = Solution()
    list = []
    s.xian(root, 0, list)
    print(list)
