


def list_alignment(ls, value):
    """
    # Argument
        - ls: [[1,2],[1,2,3],[1,1]]
        - value: a value using for alignment
    """
    new_list = []
    original_len = []
    ls_length = list(map(lambda x: len(x), ls))
    max_length = max(ls_length)
    for item in ls:
        idx_ls = item
        len_item = len(idx_ls)
        original_len.append(len_item)
        while len_item < max_length:
            idx_ls.append(value)
            len_item = len(idx_ls)
            
        new_list.append(idx_ls)
    return new_list, original_len


def top_n(data, n=2):
    """
    # Arguments
        data {list}: [0.3,0.1,0.5,0.01]
    
    # Returns
        mask {list}: [2,0]

    """
    index = list(range(len(data)))
    new_data = list(zip(data,index))
    # range
    # new_data = list(filter(lambda x: x[0]>0 and x[0]<1, new_data))

    sorted_new_data = sorted(new_data, key=lambda x: x[0], reverse=True)
    result = list(map(lambda x:x[1], sorted_new_data[:n]))
    result = sorted(result)
    return result


if __name__ == "__main__":
    data = [[1,2],[1,2,3],[1,1]]
    new_data = list_alignment(data, 0)
    print(new_data)
