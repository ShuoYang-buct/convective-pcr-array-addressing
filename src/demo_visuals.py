"""Display helpers only: no annotation coordinates enter the inference pipeline."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def show_stage(image, title, boxes=None, grid=None, supported=None, box_size=None, save=None):
    fig, ax = plt.subplots(figsize=(12, 4.6), layout='constrained')
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    if boxes is not None:
        for item in boxes:
            x1,y1,x2,y2=item['xyxy']
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#22d3ee',linewidth=.8))
    if grid is not None:
        w,h=box_size
        for i,(x,y) in enumerate(grid):
            color='#36d399' if i in supported else '#fb923c'
            ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,fill=False,edgecolor=color,linewidth=1))
    ax.set(title=title, xlabel='x (pixels)', ylabel='y (pixels)')
    if save:
        path=Path(save);path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=160,facecolor='white')
    return fig

def completion_box_size(candidates):
    return (float(np.median([c['width'] for c in candidates])),
            float(np.median([c['height'] for c in candidates])))
