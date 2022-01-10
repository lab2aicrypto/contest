import pandas as pd
import requests
import json
from datetime import datetime
from dateutil.parser import parse


def call_api(coin: str, minute: int, count: int, to: datetime) -> pd.DataFrame:
    """
    업비트 OpenAPI 캔들 데이터 수집

    :param coin: 가상화폐 심볼 대문자 문자열. ex) KRW-BTC
    :param minute: 분 단위
    :param to: 마지막 캔들 시각. utc 시간 기준.
    :param count: 캔들 개수
    :return: 캔들 데이터
    """
    if isinstance(to, str):
        to = parse(to)

    params = {
        'market': coin,
        'count': count,
        'to': to
    }

    params_text = "&".join([f"{key}={params[key]}" for key in params.keys()])

    base_url = "https://api.upbit.com/v1"

    path_text = f"/candles/minutes/{minute}"

    api_url = f"{base_url}{path_text}?{params_text}"

    text_value = requests.get(api_url).text
    data = json.loads(text_value)

    df = pd.DataFrame(data)

    return df
