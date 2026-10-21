import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from matrix_region_consensus import _phase, _pitches, _rectangle_starts

def predict(p, cr):
    c,s=np.cos(p[2]),np.sin(p[2])
    return np.c_[cr[:,0]*p[3]*c-cr[:,1]*p[4]*s,
                 cr[:,0]*p[3]*s+cr[:,1]*p[4]*c]+p[:2]

def assign(xy,p,rows,cols,active,gate=.4):
    c,s=np.cos(p[2]),np.sin(p[2])
    uv=(xy-p[:2])@np.array([[c,-s],[s,c]])/p[3:5]
    rc=np.rint(uv).astype(int);err=np.linalg.norm(uv-rc,axis=1)
    ids=np.flatnonzero((rc[:,0]>=0)&(rc[:,0]<cols)&(rc[:,1]>=0)&(rc[:,1]<rows)&(err<gate))
    chosen={}
    for i in ids:
        col,row=rc[i]
        if not active[row,col]:continue
        k=int(row*cols+col)
        if k not in chosen or err[i]<err[chosen[k]]:chosen[k]=int(i)
    ii=np.array(list(chosen.values()),int)
    return chosen,rc[ii],err[ii]

def image_evidence(image,size):
    """Local contrast divided by local variation; bounded influence of bright debris."""
    im=np.asarray(image,dtype=np.float32)
    if im.ndim==3:im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    sigma=max(1.,size*.22)
    fine=cv2.GaussianBlur(im,(0,0),sigma)
    background=cv2.GaussianBlur(im,(0,0),sigma*3)
    variation=np.sqrt(np.maximum(cv2.GaussianBlur((im-background)**2,(0,0),sigma*3),0))
    noise=max(.5,float(np.percentile(variation,35)))
    return np.clip((fine-background)/(variation+noise),0,1)

def sample(evidence,points):
    return cv2.remap(evidence,np.float32(points[:,0,None]),np.float32(points[:,1,None]),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel()

def estimate_lattice(candidates,rows,cols,active_mask=None,image=None,use_image=True):
    rows,cols=int(rows),int(cols)
    if min(rows,cols)<1:raise ValueError('Positive layout dimensions required')
    active=np.ones((rows,cols),bool) if active_mask is None else np.asarray(active_mask,bool)
    if active.shape!=(rows,cols):raise ValueError('Active mask shape mismatch')
    fail=lambda reason:dict(ok=False,status='review',reason=reason,pairs={},version='bounded-lattice-v3')
    if not active.any():return fail('no_active_addresses')
    if min(rows,cols)==1:return fail('one_dimensional_layout_requires_anchor')
    xy=np.asarray([[c['center_x'],c['center_y']] for c in candidates],float).reshape(-1,2)
    original=np.flatnonzero(np.isfinite(xy).all(axis=1));xy=xy[original]
    if len(xy)<4:return fail('insufficient_candidates')
    sizes=np.asarray([[c.get('width',1),c.get('height',1)] for c in candidates],float)
    sizes=sizes[np.isfinite(sizes)&(sizes>0)];size=float(np.median(sizes)) if len(sizes) else 10.
    _,indices=cKDTree(xy).query(xy,k=min(13,len(xy)))
    vec=(xy[indices[:,1:]]-xy[:,None,:]).reshape(-1,2)
    vec=vec[np.linalg.norm(vec,axis=1)>max(2,size*.7)]
    if not len(vec):return fail('no_spacing')
    angles=(np.arctan2(vec[:,1],vec[:,0])+np.pi/4)%(np.pi/2)-np.pi/4
    hist,edges=np.histogram(angles,bins=61,range=(-np.pi/12,np.pi/12))
    if not hist.any():return fail('rotation_outside_prior')
    mid=(edges[np.argmax(hist)]+edges[np.argmax(hist)+1])/2
    theta=float(np.median(angles[np.abs(angles-mid)<np.deg2rad(2)]))
    if not np.isfinite(theta) or abs(theta)>=np.pi/12:
        return fail('rotation_outside_prior')
    rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    dv=vec@rot;uv=xy@rot;hx=np.abs(dv[:,0]);vy=np.abs(dv[:,1])
    pxs=_pitches(hx[(hx>size*.7)&(vy<np.maximum(2,hx*.15))],size*2)
    pys=_pitches(vy[(vy>size*.7)&(hx<np.maximum(2,vy*.15))],size*2)
    # Seed multiple nearby pitch values, because a small pitch bias accumulates
    # over long rows and changes integer correspondences before continuous fitting.
    pxs=sorted(set(round(p*f,3) for p in pxs for f in [.94,.98,1.,1.02,1.06]))
    pys=sorted(set(round(p*f,3) for p in pys for f in [.98,1.,1.02]))
    cc,rr=np.meshgrid(np.arange(cols),np.arange(rows));cr=np.c_[cc.ravel(),rr.ravel()]
    evidence=image_evidence(image,size) if image is not None and use_image else None
    seeds=[]
    for px in pxs:
      for py in pys:
        phase=np.array([_phase(uv[:,0]/px),_phase(uv[:,1]/py)])
        scaled=uv/[px,py]-phase;integer=np.rint(scaled).astype(int)
        good=np.linalg.norm(scaled-integer,axis=1)<.4
        if good.sum()<4:continue
        for sx,sy in _rectangle_starts(integer[good,0],integer[good,1],rows,cols):
            p=np.r_[((np.array([sx,sy])+phase)*[px,py])@rot.T,theta,px,py]
            pairs,_,err=assign(xy,p,rows,cols,active)
            seeds.append((len(pairs)-.1*sum(err),p))
    if not seeds:return fail('no_proposals')
    seeds.sort(key=lambda x:x[0],reverse=True)
    # Preserve several pitch families instead of fitting only the winning raw seed.
    selected=[]
    for _,p in seeds:
        if all(np.linalg.norm((p[:2]-q[:2])/p[3:5])>.3 or np.max(abs(p[3:5]/q[3:5]-1))>.025 for q in selected):selected.append(p)
        if len(selected)>=48:break
    fits=[]
    for p in selected:
        initial=p.copy()
        lower=np.r_[initial[:2]-initial[3:5],-np.pi/12,initial[3:5]*.85]
        upper=np.r_[initial[:2]+initial[3:5],np.pi/12,initial[3:5]*1.15]
        for _ in range(4):
            pairs,rc,err=assign(xy,p,rows,cols,active)
            if len(pairs)<4:break
            target=xy[list(pairs.values())]
            new=least_squares(lambda z:(predict(z,rc)-target).ravel(),p,bounds=(lower,upper),loss='soft_l1',f_scale=max(1,min(p[3:5])*.06),max_nfev=40).x
            if np.max(abs(new-p))<.01:p=new;break
            p=new
        # Reconsider neighboring integer origins after continuous refinement.
        for dy in [-1,0,1]:
          for dx in [-1,0,1]:
            pp=p.copy();pp[:2]=predict(p,np.array([[dx,dy]]))[0]
            pairs,_,err=assign(xy,pp,rows,cols,active)
            grid=predict(pp,cr)
            score=len(pairs)-.1*float(sum(err))
            visual=0.
            if evidence is not None:
                values=sample(evidence,grid)[active.ravel()]
                # Evaluate weak/undetected locations too, using clipped local contrast.
                visual=float(values.sum());score+=visual
            fits.append(dict(p=pp,pairs=pairs,score=score,visual=visual,err=err))
    fits.sort(key=lambda x:x['score'],reverse=True)
    best=fits[0];p=best['p'];grid=predict(p,cr);pairs=best['pairs']
    alternative=next((f for f in fits[1:] if np.median(np.linalg.norm(predict(f['p'],cr)-grid,axis=1))>min(p[3:5])*.45),None)
    gap=float(best['score']-alternative['score']) if alternative is not None else None
    coverage=len(pairs)/int(active.sum());p95=float(np.percentile(best['err'],95)) if len(pairs) else 1.
    reasons=[]
    if coverage<.6:reasons.append('low_detection_support')
    if p95>.2:reasons.append('large_residual')
    if gap is None or gap<2:reasons.append('ambiguous_index_or_pitch')
    # Require observed support on all four outer active boundaries. Image evidence
    # may rank diagnostics, but must not silently certify an unseen boundary.
    ar,ac=np.nonzero(active);keys=np.array(list(pairs),int)
    if not len(keys) or any(not np.any(v==edge) for v,edge in [(keys//cols,ar.min()),(keys//cols,ar.max()),(keys%cols,ac.min()),(keys%cols,ac.max())]):reasons.append('unobserved_boundary')
    return dict(ok=True,version='bounded-lattice-v3',status='review' if reasons else 'accept',
                review_reasons=reasons,grid=grid.tolist(),pitch=p[3:5].tolist(),rotation_deg=float(np.rad2deg(p[2])),
                pairs={str(k):int(original[v]) for k,v in pairs.items()},support_fraction=coverage,
                normalized_residual_p95=p95,alternative_score_gap=gap,image_score=best['visual'],
                proposals=len(seeds),refined=len(selected),advisory_only=True)
