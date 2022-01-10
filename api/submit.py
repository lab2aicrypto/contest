import requests
from datetime import datetime, timedelta


def add_feed(mentor_page_id: str, coin: str, buy: float, sell: float,
             predict_minute_range: int, start_time: datetime):
    """
    실시간 가격 예측 결과 제출

    :param mentor_page_id: 참가 팀 별 id
    :param coin: 예측한 가상화폐의 심볼 대문자 문자열. ex) KRW-BTC
    :param buy: 매수가
    :param sell: 매도가
    :param predict_minute_range: 예측 길이
    :param start_time: 예측 시작 시간. kst 시간 기준.
    :return:
    """
    start_time = start_time.replace(second=0, microsecond=0)
    end_time = start_time + timedelta(minutes=predict_minute_range)

    params = {
        'mentorPageId': mentor_page_id,
        'market': coin,
        'startPrice': buy,
        'targetPrice': sell,
        'validateMin': predict_minute_range,
        'startTime': start_time
    }

    requests.post("http://api.cosign.cc/api/sign/add", data=params)

    print(f"피드 추가 {params}")
