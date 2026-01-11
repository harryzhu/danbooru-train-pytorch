# coding = utf-8
import os
import torch
from PIL import Image, ImageDraw
import numpy as np

from TorchDeepDanbooru import deep_danbooru_model

import time

APP_ROOT = os.path.dirname(__file__)

model_dan = deep_danbooru_model.DeepDanbooruModel()
model_dan.load_state_dict(torch.load(f'{APP_ROOT}/output/DeepDanbooruModel/model.pth'))

model_dan.eval()
model_dan.half()
model_dan.cuda()


def danbooru_tags(fpath):
    tags = []
    try:
        pic = Image.open(fpath).convert("RGB").resize((512, 512))

        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad(), torch.autocast("cuda"):
            x = torch.from_numpy(a).cuda()
            y = model_dan(x)[0].detach().cpu().numpy()
            for n in range(1):
                model_dan(x)
        for i, p in enumerate(y):
            print(model_dan.tags[i], "%.16f" % p)
            if p >= 0.3:
                #print(model_dan.tags[i], p)
                tags.append(model_dan.tags[i])

        return tags
    except Exception as err:
        print(f'ERROR.danbooru_tags.20: {err}')
        return []




t = danbooru_tags(f"{APP_ROOT}/002.png")

print("-"*20)
print(t)



