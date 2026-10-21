"""One-sequence raw-image → YOLO → complete-grid demonstration."""
from pathlib import Path
import argparse, csv, json, sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from preprocessing import enhance_pipeline1_basic
from detector import detect_candidate_stages
from lattice_registration import estimate_lattice
from demo_visuals import show_stage, completion_box_size

def run(weights, out=None):
    import matplotlib.pyplot as plt
    out=Path(out) if out else ROOT/'outputs/demo';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((ROOT/'demo/N008/metadata.json').read_text())
    frames=[cv2.imdecode(np.fromfile(ROOT/'demo/N008'/f,np.uint8),cv2.IMREAD_GRAYSCALE) for f in meta['frames']]
    if any(f is None for f in frames): raise ValueError('Cannot read a demo frame')
    x,y,w,h=meta['crop_xywh']
    crop=lambda image:image[y:y+h,x:x+w]
    projection=np.maximum.reduce(frames)
    reference=crop(enhance_pipeline1_basic(projection))
    stages=detect_candidate_stages(reference,weights)
    candidates=stages['filtered']
    result=estimate_lattice(candidates,meta['rows'],meta['cols'],image=reference)
    if not result.get('ok'): raise RuntimeError(result)
    grid=np.array(result['grid']);supported={int(i) for i in result['pairs']}
    final=show_stage(reference,f"Complete address grid: {len(grid)} positions; {len(supported)} supported, {len(grid)-len(supported)} completed",grid=grid,supported=supported,box_size=completion_box_size(candidates),save=out/'06_complete_grid.png');plt.close(final)
    for filename,image,title,boxes in [
        ('01_source_frame.png',crop(frames[0]),'Original acquired frame (display crop)',None),
        ('02_maximum_projection.png',crop(projection),'Temporal maximum projection (10 selected frames)',None),
        ('03_enhanced_reference.png',reference,'Background subtraction and contrast enhancement',None),
        ('04_yolo_boxes.png',reference,f"YOLO boxes, confidence ≥ 0.25: {len(stages['raw'])}",stages['raw']),
        ('05_filtered_candidates.png',reference,f"Filtered candidates: {len(candidates)}",candidates)]:
        fig=show_stage(image,title,boxes=boxes,save=out/filename);plt.close(fig)
    for filename,image in [('source_crop.png',crop(frames[0])),('projection_crop.png',crop(projection)),('enhanced_reference.png',reference)]:
        cv2.imencode('.png',image)[1].tofile(out/filename)
    (out/'prediction.json').write_text(json.dumps(result,indent=2))
    with (out/'addresses.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['row','column','center_x','center_y','candidate_supported'])
        for i,(gx,gy) in enumerate(grid):writer.writerow([i//meta['cols']+1,i%meta['cols']+1,gx,gy,i in supported])
    summary={'sequence':meta['sequence'],'selected_raw_frames':len(frames),'yolo_boxes':len(stages['raw']),'filtered_candidates':len(candidates),'output_addresses':len(grid),'candidate_supported_addresses':len(supported),'completed_addresses':len(grid)-len(supported),'qc_status':result['status'],'qc_reasons':result['review_reasons'],'manual_coordinates_used':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--weights',type=Path,default=ROOT/'weights/best.pt');parser.add_argument('--output',type=Path)
    args=parser.parse_args();run(args.weights,args.output)
