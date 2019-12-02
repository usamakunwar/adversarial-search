import sys

dict = {}


for apple in range(1, 1000000):
    dict[apple] = 1


print(len(dict))
print(sys.getsizeof(dict))