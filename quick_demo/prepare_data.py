import re
import subprocess
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "datasets"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DATASETS = [
    ("ImageNet-1K train", "ImageNet2012/ILSVRC/Data/CLS-LOC/train", "ILSVRC2012_img_train.tar",
     "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"),
    ("ImageNet-1K validation", "ImageNet2012/ILSVRC/Data/CLS-LOC/val", "ILSVRC2012_img_val.tar",
     "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"),
    ("NINCO", "NINCO", "NINCO_all.tar.gz",
     "https://zenodo.org/records/8013288/files/NINCO_all.tar.gz?download=1"),
    ("ImageNet-OOD", "imagenetood", "imagenetood.tar.gz", None),
]


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    subprocess.run([
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--continue-at", "-", "--retry", "3", "--output", str(partial), url,
    ], check=True)
    partial.replace(destination)


def extract_train(archive, destination):
    with tarfile.open(archive, mode="r|") as outer:
        for member in outer:
            if not member.isfile() or not re.fullmatch(r"n\d{8}\.tar", member.name):
                raise ValueError("Unexpected training archive entry: " + member.name)
            class_dir = destination / Path(member.name).stem
            if (class_dir / ".extracted").exists():
                continue
            class_dir.mkdir(exist_ok=True)
            with outer.extractfile(member) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as inner:
                    inner.extractall(class_dir, filter="data")
            (class_dir / ".extracted").touch()


def prepare_dataset(title, folder, archive_name, url):
    root = DATA_ROOT / folder
    marker = root / ".extracting"
    if root.is_dir() and not marker.exists():
        if any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in root.rglob("*")):
            print(title + ": found.")
            return
    archive = DATA_ROOT / archive_name
    if not archive.is_file():
        if input("Download " + title + "? [y/N]: ").strip().lower() not in {"y", "yes"}:
            return
        if not url:
            raise ValueError("URL not provided for " + title + ". Please download it manually.")
        download(url, archive)
    print("Extracting " + archive_name + "...", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    marker.touch()
    if archive_name == "ILSVRC2012_img_train.tar":
        extract_train(archive, root)
    else:
        with tarfile.open(archive) as source:
            source.extractall(root, filter="data")
    marker.unlink()


def main():
    for dataset in DATASETS:
        try:
            prepare_dataset(*dataset)
        except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as error:
            print(dataset[0] + ": " + str(error))


if __name__ == "__main__":
    main()
