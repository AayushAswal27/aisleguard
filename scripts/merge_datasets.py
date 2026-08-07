from pathlib import Path
import shutil, hashlib, yaml

ROOT=Path("data")
DEST=ROOT/"merged"
IMAGE_EXTS=[".jpg",".jpeg",".png"]
FINAL_CLASSES={"person":0,"forklift":1}
SOURCES = [
    {
        "name": "loco",
        "path": ROOT / "raw" / "loco",
        "keep": ["forklift"],
    },
    {
        "name": "forklift_1",
        "path": ROOT / "raw" / "forklift_1",
        "keep": ["forklift"],
    },
    {
        "name": "worker",
        "path": ROOT / "raw" / "worker_safety",
        "keep": ["person"],
    },
    {
        "name": "logistics",
        "path": ROOT / "raw" / "logistics",
        "keep": ["person", "forklift"],
    },
    {
        "name": "vehicle",
        "path": ROOT / "raw" / "vehicle",
        "keep": ["person", "forklift"],
    },
]
ID_TO_NAME={v:k for k,v in FINAL_CLASSES.items()}

def load_class_mapping(dataset_path:Path)->dict[str,int]:
    with open(dataset_path/"data.yaml") as f:
        data=yaml.safe_load(f)
    names=data["names"]
    if isinstance(names,dict):
        return {v:int(k) for k,v in names.items()}
    return {n:i for i,n in enumerate(names)}

def find_image(img_dir:Path,stem:str):
    for ext in IMAGE_EXTS:
        p=img_dir/f"{stem}{ext}"
        if p.exists():
            return p
    return None

def prepare():
    if DEST.exists():
        shutil.rmtree(DEST)
    for split in ["train","valid","test"]:
        (DEST/split/"images").mkdir(parents=True,exist_ok=True)
        (DEST/split/"labels").mkdir(parents=True,exist_ok=True)

def main():
    prepare()
    total_images=0
    total_labels=0
    per_class={k:0 for k in FINAL_CLASSES}
    print("="*60)
    print("AISLEGUARD DATASET MERGER")
    print("="*60)
    for source in SOURCES:
        print(f"\nProcessing {source['name']}...")
        cmap=load_class_mapping(source["path"])
        keep={}
        for cname in source["keep"]:
            if cname in cmap:
                keep[cmap[cname]]=FINAL_CLASSES[cname]
                print(f"  {cname}: {cmap[cname]} -> {FINAL_CLASSES[cname]}")
        for split in ["train","valid","test"]:
            ldir=source["path"]/split/"labels"
            idir=source["path"]/split/"images"
            if not ldir.exists():
                continue
            for lab in sorted(ldir.glob("*.txt")):
                out=[]
                with open(lab) as f:
                    for line in f:
                        parts=line.strip().split()
                        if len(parts)!=5:
                            continue
                        cls=int(parts[0])
                        if cls in keep:
                            parts[0]=str(keep[cls])
                            out.append(" ".join(parts))
                if not out:
                    continue
                img=find_image(idir,lab.stem)
                if img is None:
                    continue
                hid=hashlib.md5(f"{source['name']}_{split}_{lab.stem}".encode()).hexdigest()[:12]
                new=f"{source['name']}_{hid}"
                shutil.copy2(img,DEST/split/"images"/(new+img.suffix))
                with open(DEST/split/"labels"/(new+".txt"),"w") as f:
                    f.write("\n".join(out))
                total_images+=1
                total_labels+=len(out)
                for l in out:
                    per_class[ID_TO_NAME[int(l.split()[0])]]+=1
    with open(DEST/"data.yaml","w") as f:
        yaml.safe_dump({
            "path":str(DEST.resolve()),
            "train":"train/images",
            "val":"valid/images",
            "test":"test/images",
            "nc":2,
            "names":["person","forklift"]
        },f,sort_keys=False)
    print("\n"+"="*60)
    print("MERGE COMPLETE")
    print("="*60)
    print(f"Images copied : {total_images}")
    print(f"Labels written: {total_labels}")
    print("\nPer-class:")
    for k,v in per_class.items():
        print(f"  {k}: {v}")
if __name__=="__main__":
    main()
