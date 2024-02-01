from utils.parser import Parser

parser = Parser()
sbs = parser.raw_to_df(['files/0_500_pickscore_coco'], do_overlap=True)
aggr = parser.aggregate(sbs)
print(aggr)

#url = 'https://storage.yandexcloud.net/yandex-research/new_adaptive_sbs/coco_600/diversity/cd5/seed600_1200/185.jpg'
#response = requests.get(url)
#img = Image.open(BytesIO(response.content))
