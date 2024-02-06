import argparse
from main import get_args_parser
from models.deformable_detr import build
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import requests
import matplotlib.pyplot as plt
torch.set_grad_enabled(False)

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def input_image():
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    # mean-std normalize the input image (batch-size: 1)
    trans_img = transform(im).unsqueeze(0)
    return im, trans_img



def inference(model_path):
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    model_dict = torch.load(model_path)["model"]
    model, _, post_process = build(args)
    model.load_state_dict(model_dict)
    im, trans_img = input_image()
    img_w, img_h = im.size
    target_size = torch.tensor([img_h, img_w])[None]

    model.eval()
    model.to("cuda")
    trans_img = trans_img.float().to("cuda")
    target_size = target_size.to("cuda")
    ret = model(trans_img)
    results = post_process["bbox"](ret, target_size)
    score, labels, bboxes = results[0]["scores"], results[0]["labels"], results[0]["boxes"]
    mask = score > 0.5
    score = score[mask]
    labels = labels[mask]
    bboxes = bboxes[mask, :]
    for i in range(len(score)):
        obj_score = score[i]
        obj_label = labels[i]
        xmin, ymin, xmax, ymax = bboxes[i, :].cpu().numpy()
        plt.imshow(im)
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(xmin, ymin, f"{obj_label}-{obj_score:.2f}", bbox=dict(facecolor='green', alpha=0.5))
    plt.show()


if __name__ == "__main__":
    model_path = "/home/minivision/Model/LLM/r50_deformable_detr-checkpoint.pth"
    inference(model_path)