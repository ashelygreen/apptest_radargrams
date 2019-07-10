from fastai import *
from fastai.vision import *


def l1_loss(input, tar_regr, tar_clsf):
    
    # mask inputs and targets to ignore bounding boxes for non-graves
    mask = tar_clsf[:,:,None].float().repeat(1, 1, 4) 
    inp_masked = input[0] * mask
    tar_masked = tar_regr * mask
    return F.l1_loss(inp_masked, tar_masked, reduction='mean')

def cross_entropy(input, tar_regr, tar_clsf): return F.binary_cross_entropy(input[1], tar_clsf.float())

def accuracy(input, tar_regr, tar_clsf): return ((input[1] > 0.5).int() == tar_clsf.int()).double().mean()
    

class MyLoss(nn.Module):
    def forward(self, input, tar_regr, tar_clsf):
        tar_clsf[tar_clsf==2] = 0
        loss_regr = l1_loss(input, tar_regr, tar_clsf)
        loss_clsf = cross_entropy(input, tar_regr, tar_clsf)
        return loss_regr + loss_clsf

    
class ConvBNDrop(nn.Module):
    def __init__(self, nin, nout, stride=2, drop=0.1):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(drop)

    def forward(self, x): 
        return self.drop(self.bn(F.relu(self.conv(x))))
    #         return self.drop(self.bn(F.leaky_relu(self.conv(x)))) 


class MyHead(nn.Module):   
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.1)
        self.sconv0 = ConvBNDrop(2048, 1024, stride=1)
        self.sconv1 = ConvBNDrop(1024, 512, stride=1)
        self.sconv2 = ConvBNDrop(512, 256)
        self.oconv_regr = nn.Conv2d(256, 4, 3, stride=1, padding=0)
        self.oconv_clsf = nn.Conv2d(256, 1, 3, stride=1, padding=0)
        
    def forward(self, x):
        x = self.drop(F.relu(x))
        x = self.sconv0(x)
        x = self.sconv1(x)
        x = self.sconv2(x)
        
        o_regr = torch.tanh(self.oconv_regr(x))
        o_clsf = torch.sigmoid(self.oconv_clsf(x))
        
        return [o_regr.squeeze(dim=-1).squeeze(dim=-1)[:,None,:], o_clsf.squeeze(dim=-1).squeeze(dim=-1)]

def reconstruct(self, t, x):
    (bboxes, labels) = t
    if len((labels - self.pad_idx).nonzero()) == 0: return
    i = (labels - self.pad_idx).nonzero().min()
    bboxes,labels = bboxes[i:],labels[i:]
#     return ImageBBox.create(*x.size, bboxes, labels=labels, classes=self.classes, scale=False)
    return ImageBBox.create(*x.size, bboxes, labels=[int(labels>0.5)], classes=self.classes, scale=False)

def predict(self, item:ItemBase, return_x:bool=False, batch_first:bool=True, with_dropout:bool=False, **kwargs):
    "Return predicted class, label and probabilities for `item`."
    batch = self.data.one_item(item)
    res = self.pred_batch(batch=batch, with_dropout=with_dropout)
    raw_pred,x = grab_idx(res,0,batch_first=batch_first),batch[0]
    norm = getattr(self.data,'norm',False)
    if norm:
        x = self.data.denorm(x)
        if norm.keywords.get('do_y',False): raw_pred = self.data.denorm(raw_pred)
    ds = self.data.single_ds
    pred = ds.y.analyze_pred(raw_pred, **kwargs)
    x = ds.x.reconstruct(grab_idx(x, 0))
#     y = ds.y.reconstruct(pred, x) if has_arg(ds.y.reconstruct, 'x') else ds.y.reconstruct(pred)
    y = reconstruct(ds.y, pred, x) 
    return (x, y, pred, raw_pred) if return_x else (y, pred, raw_pred)


# ImageBBox.show
def show(self, y:Image=None, ax:plt.Axes=None, figsize:tuple=(3,3), title:Optional[str]=None, hide_axis:bool=True, color:str='white', noshow=False, **kwargs):
    "Show the `ImageBBox` on `ax`."
    if ax is None and not noshow: _,ax = plt.subplots(figsize=figsize)
    bboxes, lbls = self._compute_boxes()
    h,w = self.flow.size
    bboxes.add_(1).mul_(torch.tensor([h/2, w/2, h/2, w/2])).long()
    for i, bbox in enumerate(bboxes):
        if lbls is not None: text = str(lbls[i])
        else: text=None
        if noshow:
            return bb2hw(bbox), text
        _draw_rect(ax, bb2hw(bbox), text=text, color=color)                   

def _draw_rect(ax:plt.Axes, b:Collection[int], color:str='white', text=None, text_size=14):
    "Draw bounding box on `ax`."
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    _draw_outline(patch, 4)
    if text is not None:
        patch = ax.text(*b[:2], text, verticalalignment='top', color=color, fontsize=text_size, weight='bold')
        _draw_outline(patch,1)

def _draw_outline(o:Patch, lw:int):
    "Outline bounding box onto image `Patch`."
    o.set_path_effects([patheffects.Stroke(linewidth=lw, foreground='black'), patheffects.Normal()])