from utils.parser import Parser

parser = Parser()
sbs = parser.raw_to_df(['files/coco_diversity_teacher_50_executionResult.out'], overlap=3)
aggr = parser.aggregate(sbs)
print(aggr)

#url = 'https://storage.yandexcloud.net/yandex-research/new_adaptive_sbs/coco_600/diversity/cd5/seed600_1200/185.jpg'
#response = requests.get(url)
#img = Image.open(BytesIO(response.content))
