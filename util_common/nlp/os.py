import os
import chardet
import warnings

import pandas as pd 

def read_content(dir_file):
    """ 
    Read the file as bite
    and return the content
    
    Arguments:
        dir_file {[str]} -- [description]
    
    Returns:
        [str] -- [description]
    """

    with open(dir_file,"rb") as rb:
        content = rb.read()
    
    encoder_code = chardet.detect(content)["encoding"]
    try:
        content = content.decode(encoder_code)
    except:
        message = "This file code {} is error, and ignored".format(dir_file)
        warnings.warn(message)
        content = content.decode(encoder_code, "ignore")
    return content


def read_excel(file):
    """
    Argument
        # file {str}
    
    Return 

    """
    data = pd.read_excel(file)
    return data

def read_folder_content(dir_folder):
    """[summary]
    
    Arguments:
        dir_folder {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    total_content = []
    list_files = os.listdir(dir_folder)
    for dir_file in list_files:
        content = read_content(dir_file=dir_folder+dir_file)
        total_content.append(content)
    return total_content


def list_excel(columns, data, file_name):
    """
    # Arguments
        - columns {list}: ["a","b"]
        - data {list}:[["a-v","b-v"]["a-v1","a-v2"]]
    """
    df = pd.DataFrame(columns=columns,data=data)
    writer = pd.ExcelWriter(file_name+'.xlsx')
    df.to_excel(writer,'Sheet1')
    return "Done"
