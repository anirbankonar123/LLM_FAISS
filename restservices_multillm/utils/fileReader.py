
import io
import openpyxl
from itertools import islice
import pandas as pd
import shutil
import csv

def read_excel(contents):

    xlsx = io.BytesIO(contents)
    wb = openpyxl.load_workbook(xlsx)
    ws = wb.active

    data = ws.values
    cols = next(data)[1:]
    data = list(data)
    idx = [r[0] for r in data]
    data = (islice(r, 1, None) for r in data)
    df = pd.DataFrame(data, index=idx, columns=cols)

    return df

def read_csv(file,filepath):
    with open(filepath, "wb", ) as buffer:
        shutil.copyfileobj(file.file, buffer)
    buffer.close()

    # Read uploaded file as dataframe to validate
    with open(filepath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        requirement_data = list(csvReader)
    csvf.close()

    df = pd.DataFrame(requirement_data)

    return df


def read_doc(file, filepath: str) -> str:
    """Read the document bytestream and return the copied file location"""
    try:
        with open(
                filepath,
                "wb",
        ) as buffer:
            shutil.copyfileobj(file.file, buffer)

    except Exception as exc:
        return (str(exc))
    finally:
        buffer.close()

    return filepath