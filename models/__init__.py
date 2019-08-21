#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import timm
from base import BaseModel


#------------------------------------------------------------------------------
#  ClassificationModel
#------------------------------------------------------------------------------
class ClassificationModel(BaseModel):
    def __init__(self, model_name, pretrained=False, num_classes=1000,
                in_chans=3, checkpoint_path='', **kwargs):

        model = timm.create_model(model_name, pretrained, num_classes, in_chans, checkpoint_path, **kwargs)
        self.__class__ = type(model.__class__.__name__, (self.__class__, model.__class__), {})
        self.__dict__ = model.__dict__

    def forward(self, images, **kargs):
        output = super(ClassificationModel, self).forward(images)
        return {"logits": output}
