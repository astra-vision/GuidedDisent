import torch
import yaml
from MUNIT.model_infer import MUNIT_infer
import yaml
from droprenderer import DropModel
import gradio as gr
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def init_model():
    hyperparameters = get_config('configs/params_net.yaml')
    model = MUNIT_infer(hyperparameters)
    weights = torch.load('weights/pretrained.pth')
    model.gen_a.load_state_dict(weights['a'])
    model.gen_b.load_state_dict(weights['b'])
    model = model.to(device)
    return model

def apply_fn(*inputs):
    input_image = inputs[0]

    img_resize_width = inputs[1]
    img_resize_height = inputs[2]

    input_image = input_image.resize((img_resize_width, img_resize_height), Image.BILINEAR)

    drops_size = inputs[3]
    drops_frequency = inputs[4]

    drops_shape = inputs[5]
    drops_sigma = torch.zeros(1).fill_(inputs[6]).to(device)

    dm = DropModel(size_threshold=1 / float(drops_size), frequency_threshold=drops_frequency, shape_threshold=drops_shape)
    model = init_model()

    im = ToTensor()(input_image) * 2 - 1
    im = im.unsqueeze(0).cuda()
    im_res = model.forward(im)
    im_drops = dm.add_drops(im_res, sigma=drops_sigma)

    im_res = ToPILImage()((im_res[0].cpu() + 1) / 2)
    im_drops = ToPILImage()((im_drops[0].cpu() + 1) / 2)

    return im_res, im_drops

def build_app():
    # parameters
    img_shape = [
        gr.Number(value=400, label='resize_width'),
        gr.Number(value=224, label='resize_hight'),
    ]

    size_builder = [
        gr.Slider(1, 150, value=50, label="drop_size", info="Increasing this parameter increases the drops size"),
    ]
    frequency_builder = [
        gr.Slider(3, 100, value=8, label="drop_frequency", info="Increasing this parameter increases the drops number."),
    ]
    shape_builder = [
        gr.Slider(0, 2, value=0.6, step=0.01, label="drop_shape", info="Increasing this parameter makes drops rounder, but smaller")
    ]
    sigma_builder = [
        gr.Slider(1, 50, value=4, step=1, label="drop_sigma", info="Defocus blur kernel size.")
    ]

    # input image
    default_image = Image.open('sample_nuscenes.png')
    image_builder = [gr.Image(type='pil', value=default_image)]

    builder = image_builder + img_shape + size_builder + frequency_builder + shape_builder + sigma_builder

    output_builder = [
        gr.Image(type='pil', label='wet', info='This image should have no drops but look wet'),
        gr.Image(type='pil', label='rainy', info='This image should look rainy, with wetness and drops')
    ]

    with open('heading.md') as file:
        heading = file.read()

    gr.Interface(fn=apply_fn, inputs=builder, outputs=output_builder, title='Physics-aware guided disentanglement demo', description=heading).launch(server_port=8899)


if __name__ == '__main__':
    build_app()
