import requests
from datetime import datetime, timedelta


def add_feed(mentor_page_id: str, username: str, password: str, coin: str, buy: float, sell: float,
             predict_minute_range: int, start_time: datetime):
    """
    실시간 가격 예측 결과 제출

    :param password:
    :param username:
    :param mentor_page_id: 참가 팀 별 id
    :param coin: 예측한 가상화폐의 심볼 대문자 문자열. ex) KRW-BTC
    :param buy: 매수가
    :param sell: 매도가
    :param predict_minute_range: 예측 길이 (분)
    :param start_time: 예측 시작 시간. kst 시간 기준.
    :return:
    """
    end_time = start_time + timedelta(minutes=predict_minute_range)

    params = {
        "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "importanceValue": 0,
        "market": coin,
        "mentorPageId": mentor_page_id,
        "password": password,
        "startPrice": buy,
        "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "targetPrice": sell,
        "username": username,
        "validateMin": predict_minute_range
    }

    response = requests.post("http://partner-api.cosign.cc/api/sign/add", data=params)

    if response.ok:
        print(f"response: {response}\n"
              f"피드 추가: {params}")
    else:
        print(f"response: {response}\n"
              f"피드 추가 실패: {params}")
