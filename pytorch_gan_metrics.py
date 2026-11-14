# pytorch_gan_metrics.py
# Compatibility shim: project expects old name `pytorch_gan_metrics`
from pytorch_image_generation_metrics import (
    get_inception_score_and_fid,
    get_fid,
    get_inception_score,
)
