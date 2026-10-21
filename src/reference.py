import numpy as np
from preprocessing import enhance_pipeline1_basic
def selected_indices(frame_count):
    if frame_count<=10:return list(range(frame_count))
    earlier=list(range(frame_count-10));later=list(range(frame_count-10,frame_count))
    return [earlier[i] for i in np.linspace(0,len(earlier)-1,min(5,len(earlier))).round().astype(int)]+[later[i] for i in np.linspace(0,9,5).round().astype(int)]
def build_reference(original_frames,crop=None):
    """Call only after acquisition. crop=(x,y,width,height) is optional."""
    ids=selected_indices(len(original_frames));projection=np.maximum.reduce([original_frames[i] for i in ids])
    enhanced=enhance_pipeline1_basic(projection)
    if crop is not None:
        x,y,w,h=map(int,crop);enhanced=enhanced[y:y+h,x:x+w]
    return enhanced
