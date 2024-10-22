# 打开本地的data/news.json文件
def trim_news_data(length: int = 100):
    """
    从`news.json`文件中截取指定长度的数据记录，并保存到`news.json`文件中
    :param length: 截取长度
    """
    import json

    with open('data/news.json', 'r') as file:
        data = json.load(file)
    dict_list = data['data'][:length]
    data['data'] = dict_list
    with open('data/news.json', 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)