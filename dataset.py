import requests

from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset
from io import BytesIO

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def url_to_img(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def open_img(path):
    img = Image.open(path)
    return img

class DiversityDataset(Dataset):

    def __init__(self, df, local_path=None, preprocess=None):
        if preprocess is None:
            preprocess = _transform(224)
        if local_path is None:
            print('Dataset will be loaded from url')
        else:
            print(f'Dataset downloaded locally in {local_path}')

        self.local_path = local_path

        self.preprocess = preprocess
        self.df = df
        self.data = self.make_data()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.df)

    def make_data(self):
        # Can be reimplemented further
        print('Creating dataset...')

        list_of_dicts = self.df.to_dict('records')

        # Make images from urls
        for j, item in tqdm(enumerate(list_of_dicts)):
            # img1
            img_1_url = item['image_1']
            if self.local_path is not None:
                splitted_path = img_1_url.split('/')
                main_path = f'{splitted_path[-3]}/{splitted_path[-2]}/{splitted_path[-1]}'
                path = f'{self.local_path}/{main_path}'
                pil_image = open_img(path)
            else:
                pil_image = url_to_img(img_1_url)

            image_1 = self.preprocess(pil_image)
            list_of_dicts[j]['image_1'] = image_1

            # img2
            img_2_url = item['image_2']
            if self.local_path is not None:
                splitted_path = img_2_url.split('/')
                main_path = f'{splitted_path[-3]}/{splitted_path[-2]}/{splitted_path[-1]}'
                path = f'{self.local_path}/{main_path}'
                pil_image = open_img(path)
            else:
                pil_image = url_to_img(img_2_url)

            image_2 = self.preprocess(pil_image)
            list_of_dicts[j]['image_2'] = image_2

        return list_of_dicts
