import requests
from datetime import datetime


def add_feed(username: str, password: str, coin: str, buy: float, sell: float,
             predict_minute_range: int, start_time: datetime):
    """
    실시간 가격 예측 결과 제출

    :param username: 참가팀 별 유저 이름
    :param password: 참가팀 별 비밀번호
    :param coin: 예측한 가상화폐의 심볼 대문자 문자열. ex) KRW-BTC
    :param buy: 매수가
    :param sell: 매도가
    :param predict_minute_range: 예측 길이 (분)
    :param start_time: 예측 시작 시간. kst 시간 기준.
    :return:
    """
    params = {
        "username": username,
        "password": password,
        "market": coin,
        "priceStart": buy,
        "priceTarget": sell,
        "timeStart": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "timeValidateMinutes": predict_minute_range
    }

    response = requests.post("http://partner-api.cosign.cc/api/sign/v2/add", data=params)

    if response.ok:
        print(f"response: {response.json()}\n"
              f"피드 추가: {params}")
    else:
        print(f"response: {response.json()}\n"
              f"피드 추가 실패: {params}")
