
from timm.models import create_model




# def build_pvt_tiny(num_classes):
#     return create_model(
#                 "twins_pcpvt_small",
#                 False,
#                 num_classes,
#                 0.0,
#                 0.5,
#                 None,
#          )

cfg = dict(
    model='pvt_tiny',
    drop_path=0.1,
    clip_grad=None,
    output_dir='checkpoints/pvt_tiny',
)