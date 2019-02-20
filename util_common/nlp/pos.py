import requests
import jieba.posseg as psg

def pos_japanese(text):
    """ Call a japanese pos api

    # Arguments
        text {str}

    # Returns
        list_pos {list}: [('対立', '名詞-サ変接続'), ('の', '助詞-連体化'),]
        
    """
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    result = requests.post("http://192.168.2.130:8787", data={"language": text},headers=headers)
    data = eval(result.text)
    def preprocessing(data):
        data = data["data"][0].split("\n")[:-1]
        ls_pos = []
        for item in data:
            num_values = len(item.split("\t"))
            if num_values >= 3:
                ls_pos.append((item.split("\t")[0], item.split("\t")[3]))
        return ls_pos
    list_pos = preprocessing(data)
    return list_pos


def pos_chinese(text):
 
    seg = psg.cut(text)
    ls_pos = []
    for ele in seg:
        ls_pos.append((ele.word, ele.flag))
    return ls_pos

if __name__ == "__main__":
    # text = "対立の激化が懸念されていた。"
    # pos_japanese(text)
    text = u"我和王非去北京大学玩"
    print(pos_chinese(text))
