"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# import pytz
import sys
#!/home/azureuser/anaconda3/envs/myenv2/bin/python
# import sys
# sys.path.append('/home/azureuser/anaconda3/envs/myenv2/bin/pip')

import os
import re
import time
import shutil
from uuid import uuid4
from functools import partial
from http import HTTPStatus
from fastapi import FastAPI, File, UploadFile, Form
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pypdfium2
import torch
from nougat import NougatModel
from nougat.postprocessing import markdown_compatible, close_envs
from nougat.utils.dataset import ImageDataset
from nougat.utils.checkpoint import get_checkpoint
from nougat.dataset.rasterize import rasterize_paper
from nougat.utils.device import move_to_device
from tqdm import tqdm
from utils import get_mongo_collection
from dotenv import load_dotenv
from latext import latex_to_text

load_dotenv()

nougat_pages = get_mongo_collection('nougat_pages')
index_name = "index_bookId"
indexes_info = nougat_pages.list_indexes()
index_exists = any(index_info["name"] == index_name for index_info in indexes_info)
if not index_exists:
    nougat_pages.create_index("bookId", name=index_name, background=True)

model = None
SAVE_DIR = Path("../pdfs")
BATCHSIZE = 6
NOUGAT_CHECKPOINT = get_checkpoint()
print("NOUGAT_CHECKPOINT >>> ", NOUGAT_CHECKPOINT)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model, BATCHSIZE, NOUGAT_CHECKPOINT
    if model is None:
        model = NougatModel.from_pretrained(NOUGAT_CHECKPOINT)
        model = move_to_device(model, cuda=BATCHSIZE > 0)
        if BATCHSIZE <= 0:
            BATCHSIZE = 1
        model.eval()
    yield
    # Clean up the ML models and release the resources
    del model
    torch.cuda.empty_cache()

app = FastAPI(title="Nougat API", lifespan=lifespan)
origins = ["http://localhost", "http://127.0.0.1"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def root():
    """Health check."""
    response = {
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict/")
async def predict(
    bookId: str = Form(...),
    page_nums: list[str] = Form(...),
    file: UploadFile = File(...), start: int = None, stop: int = None, skipping: bool = False
) -> str:
    """
    Perform predictions on a PDF document and return the extracted text in Markdown format.

    Args:
        file (UploadFile): The uploaded PDF file to process.
        bookId (str): The bookId of the book
        bookname (str): The name of the book
        start (int, optional): The starting page number for prediction.
        stop (int, optional): The ending page number for prediction.
        skipping (book, optional): Whether auto detection of out of domain
            pages should be skipped or not

    Returns:
        str: The extracted text in Markdown format.
    """
    st_time = time.time()
    pdfbin = file.file.read()
    pdf = pypdfium2.PdfDocument(pdfbin)
    file_unique_id = uuid4().hex
    save_path = SAVE_DIR / f"{file_unique_id}"

    if start is not None and stop is not None:
        pages = list(range(start - 1, stop))
    else:
        pages = list(range(len(pdf)))
    predictions = [""] * len(pages)
    dellist = []
    if save_path.exists():
        for computed in (save_path / "pages").glob("*.mmd"):
            try:
                idx = int(computed.stem) - 1
                if idx in pages:
                    i = pages.index(idx)
                    print("skip page", idx + 1)
                    predictions[i] = computed.read_text(encoding="utf-8")
                    dellist.append(idx)
            except Exception as e:
                print(e)
    compute_pages = pages.copy()
    for el in dellist:
        compute_pages.remove(el)
    images = rasterize_paper(pdf, pages=compute_pages)
    global model

    dataset = ImageDataset(
        images,
        partial(model.encoder.prepare_input, random_padding=False),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCHSIZE,
        pin_memory=True,
        shuffle=False,
    )

    for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if sample is None:
            continue
        model_output = model.inference(image_tensors=sample, early_stopping=skipping)
        for j, output in enumerate(model_output["predictions"]):
            if model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    disclaimer = "\n\n+++ ==WARNING: Truncated because of repetitions==\n%s\n+++\n\n"
                else:
                    disclaimer = (
                        "\n\n+++ ==ERROR: No output for this page==\n%s\n+++\n\n"
                    )
                rest = close_envs(model_output["repetitions"][j]).strip()
                if len(rest) > 0:
                    disclaimer = disclaimer % rest
                else:
                    disclaimer = ""
            else:
                disclaimer = ""

            predictions[pages.index(compute_pages[idx * BATCHSIZE + j])] = (
                markdown_compatible(output) + disclaimer
            )

    (save_path / "pages").mkdir(parents=True, exist_ok=True)
    for idx, page_num in enumerate(pages):
        (save_path / "pages" / ("%02d.mmd" % (page_num + 1))).write_text(
            predictions[idx], encoding="utf-8"
        )
    extracted_pdf_directory = save_path
    pdfId_path = os.path.join(extracted_pdf_directory, 'pages')
    if os.path.exists(pdfId_path):
        files = sorted(os.listdir(pdfId_path))
        pattern = r'(\\\(.*?\\\)|\\\[.*?\\\])'
        extracted_nougat_pages = []
        for filename, page_num in zip(files, page_nums):
            page_equations = []
            file_path = os.path.join(pdfId_path, filename)
            latex_text = ""
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    latex_text = file.read()
                    latex_text = latex_text.replace("[MISSING_PAGE_POST]", "")
            def replace_with_uuid(match):
                equationId = uuid4().hex
                match_text = match.group()
                text_to_speech = latext_to_text_to_speech(match_text)
                page_equations.append({
                    'id': equationId,
                    'text': match_text,
                    'text_to_speech': text_to_speech
                })
                return f'{{{{equation:{equationId}}}}}'
            page_content = re.sub(pattern, replace_with_uuid, latex_text)
            page_content = re.sub(r'\s+', ' ', page_content).strip()
            page_object = {
                "page_num": page_num,
                "text": page_content,
                "tables": [],
                "figures": [],
                "page_equations": page_equations
            }
            extracted_nougat_pages.append(page_object)
            nougat_pages.find_one_and_update(
                {"bookId": bookId},
                {"$set": {f"pages.{page_num}": page_object}},
                upsert=True
            )
        shutil.rmtree(extracted_pdf_directory)
    print("Total time taken: {:.2f} seconds".format(time.time() - st_time))
    return file_unique_id

def latext_to_text_to_speech(text):
    # Remove leading backslashes and add dollar signs at the beginning and end of the text
    text = "${}$".format(text.lstrip('\\'))
    # Convert the LaTeX text to text to speech
    text_to_speech = latex_to_text(text)
    return text_to_speech


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1])
    uvicorn.run("app:app", host='0.0.0.0', port=port, reload=True)
