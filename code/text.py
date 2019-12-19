
a = [[0 for x in range(5)] for y in range(5)]

for r in range(1, 5):
    for i in range(5-r):
        a[i][i+r]=1
for x in a:
    print(x)
