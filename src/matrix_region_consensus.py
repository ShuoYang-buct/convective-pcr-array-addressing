import numpy as np

def _phase(values):
    fractions=values % 1
    histogram,edges=np.histogram(fractions,bins=32,range=(0,1))
    # Circular smoothing avoids splitting a phase across 0 and 1.
    smooth=histogram+np.roll(histogram,1)+np.roll(histogram,-1)
    center=(np.argmax(smooth)+.5)/32
    delta=(fractions-center+.5)%1-.5
    good=np.abs(delta)<.18
    return float((center+np.median(delta[good]))%1) if good.any() else center

def _pitches(values, fallback):
    v=np.asarray(values);v=v[np.isfinite(v)&(v>1)]
    if not len(v):return [float(fallback)]
    # Short repeated neighbor distances identify pitch; distant pairs do not set the range.
    low,high=np.percentile(v,[5,85]);step=max(1.,float(np.median(v))*.045)
    edges=np.arange(max(1.,low-step),high+2*step,step)
    h,e=np.histogram(v,bins=edges)
    peaks=np.argsort(h)[::-1];result=[]
    for i in peaks:
        if h[i]<max(2,int(h.max()*.2)):continue
        near=v[np.abs(v-(e[i]+e[i+1])/2)<step*1.5]
        p=float(np.median(near))
        if all(abs(p-x)>min(p,x)*.15 for x in result):result.append(p)
        if len(result)==2:break
    return result or [float(np.median(v))]

def _rectangle_starts(kx,ky,rows,cols):
    xs=np.unique(kx);ys=np.unique(ky)
    occupancy=np.zeros((len(ys),len(xs)),dtype=float)
    occupancy[np.searchsorted(ys,ky),np.searchsorted(xs,kx)]=1.
    integral=np.pad(occupancy.cumsum(0).cumsum(1),((1,0),(1,0)))
    sx=np.unique(np.r_[xs,xs-cols+1]);sy=np.unique(np.r_[ys,ys-rows+1])
    left=np.searchsorted(xs,sx);right=np.searchsorted(xs,sx+cols)
    top=np.searchsorted(ys,sy);bottom=np.searchsorted(ys,sy+rows)
    scores=integral[bottom[:,None],right]-integral[top[:,None],right]-integral[bottom[:,None],left]+integral[top[:,None],left]
    order=np.argsort(scores.ravel())[::-1][:8]
    return [(int(sx[i%len(sx)]),int(sy[i//len(sx)])) for i in order]
