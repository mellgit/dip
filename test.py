from numpy import exp, array, random, dot

def main():

    """
    
    """
    """
    создание псевдо рандомного значения 
    """
    # random.seed(1)
    # print(random.random())
    
    # random.seed(1)
    # print(random.random())
    """
    
    """
    # v1 = random.random((7, 1))
    # print(v1)
    # print(v1*2)
    # print(v1*2 - 1)

    # v2 = 2 * random.random((7, 1))
    # print(v2)

    synaptic_weights = 2 * random.random((7, 1)) - 1
    print(synaptic_weights)
    # t()


def t():
    ls = [random.randint(0,2) for i in range(7)]
    mx = []
    for i in range(8):
        mx.append([random.randint(0,2) for i in range(7)])

    for i in range(len(mx)):
        print(mx[i])

    # print(mx)

if __name__ == "__main__":

    main()

