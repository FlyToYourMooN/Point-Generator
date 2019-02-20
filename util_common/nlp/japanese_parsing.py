
import requests

def japanese_parsing(text):
    """
    """
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post("http://192.168.2.130:8002", data={"question": text},headers=headers)
    data = eval(r.text)
    data = (data["data"][0])

    def preprocessing(data):
        ls_parsing = []

        current = 49
        signal_position = current

        for item in data.split("\n"):
            item = item.replace("  "," ")
            if item.find("═") != -1: signal_position = item.find("═")
            word = item.split("═")[0].strip().split("<")[0]
            word = word.split("#")[0]
            word = word.replace("EOS","")
            if word == "": continue
            ls_parsing.append((word,signal_position))
        return ls_parsing

    return preprocessing(data)

if __name__ == "__main__":
    text = "対立の激化が懸念されていた。"
    print(japanese_parsing(text))