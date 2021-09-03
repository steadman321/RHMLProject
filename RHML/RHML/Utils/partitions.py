# for a set of classes, calc all possibel partitions 
# return the left side of those partitions
def get_leftside_of_two_partitions(allClasses):
    import itertools as it
    left_list = []
    for cls in range(1,int(len(allClasses)/2)+1):
        combis = set(it.combinations(allClasses,cls))
        for c in combis:
            left_list.append(list(c))
    return left_list