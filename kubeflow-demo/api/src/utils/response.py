import pandas as pd
from io import StringIO


def predict_response_handler(
    data: pd.DataFrame,
    output_format: str
) -> str | dict:
    if output_format == 'json':
        return data.to_dict()
    elif output_format == 'csv':
        buffer = StringIO()
        buffer.seek(0)
        data.to_csv(buffer)
        return buffer.read()
    else:
        raise NotImplementedError(output_format)
