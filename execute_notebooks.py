"""Execute each notebook in a fresh kernel using this interpreter."""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import json
import os
import sys
import nbformat
from nbclient import NotebookClient

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pattern",default="*.ipynb",help="Notebook filename glob")
args=parser.parse_args()
root=Path(__file__).resolve().parent
with TemporaryDirectory() as tmp:
    kernel=Path(tmp)/"kernels"/"similarity-lab"
    kernel.mkdir(parents=True)
    (kernel/"kernel.json").write_text(json.dumps(dict(
        argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
        display_name="Image similarity",language="python")))
    os.environ["JUPYTER_PATH"]=tmp+os.pathsep+os.environ.get("JUPYTER_PATH","")
    for path in sorted((root/"notebooks").glob(args.pattern)):
        book=nbformat.read(path,as_version=4)
        client=NotebookClient(book,kernel_name="similarity-lab",timeout=3600,
                              resources={"metadata":{"path":str(root)}})
        client.execute()
        nbformat.validate(book)
        nbformat.write(book,path)
        print(path.name,flush=True)
