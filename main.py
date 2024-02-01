from utils.parser import Parser
from dataset import DiversityDataset
from models.baseline_clip import preprocess

parser = Parser()
df = parser.raw_to_df(['files/0_500_pickscore_coco'], do_overlap=True)
dataset = DiversityDataset(df, preprocess=preprocess)

print(dataset[0])
