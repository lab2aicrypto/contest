import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse
from statsmodels.tsa.arima.model import ARIMA
from apscheduler.schedulers.blocking import BlockingScheduler

from dataset.upbit import call_api
from api.submit import add_feed


def send_prediction(mentor_page_id: str, username: str, password: str, model_fit,
                    coin: str, minute: int, predict_minute_range: int, buffer: int):
    """
    학습된 모델을 입력받아 실시간 예측을 수행하는 함수

    :param mentor_page_id: 참가팀 별 id
    :param username: 참가팀 별 유저 이름
    :param password: 참가팀 별 비밀번호
    :param model_fit: 학습된 모델
    :param coin: 코인 종류
    :param minute: 학습에 사용한 데이터의 분 봉 단위
    :param predict_minute_range: 예측 길이
    :param buffer: 예측 게시 버퍼 시간
    :return:
    """
    # 최근 캔들 하나 새로 수집
    # 데이터 수집에 사용되는 시점 파라미터(to)는 utc 시각 기준으로 사용합니다. 업비트 OpenAPI Documentation 참조.
    utc_now = datetime.utcnow().replace(second=0, microsecond=0)
    recent_candle = call_api(coin, minute, count=1, to=utc_now)
    recent_price = recent_candle["trade_price"].apply(np.log)

    # 미래 예측
    # 데이터 수집 -> 예측 -> 게시에 소요되는 시간만큼 예측을 더 길게 해야합니다. (buffer 변수 참조)
    # 예를 들어, 15분 길이 예측에서 데이터 수집 -> 예측 -> 게시까지 3분이 소요되면 18분 길이로 예측한 뒤 18분 중 처음 3분을 건너뛰고 뒤의 15분 예측 결과를 사용합니다.
    model_fit = model_fit.append(recent_price.values, refit=False)
    pred = model_fit.forecast(buffer + predict_minute_range)

    # 예측 결과로부터 매수가, 매도가를 추출합니다.
    # 로그 변환된 예측 값을 역변환하고, buffer 만큼 건너뛴 예측 결과를 사용합니다.
    pred = np.exp(pred[buffer:])
    buy_index = np.argmin(pred)
    buy = min(pred)
    sell = max(pred[buy_index:])

    # 예측 시작 시점은 kst 시각 기준입니다.
    # 예측 시작 시점은 마지막으로 수집한 데이터로부터 buffer(분) 만큼 건너뛴 시점으로 지정합니다.
    last_time = parse(recent_candle.iloc[-1]["candle_date_time_kst"])
    start_time = last_time + timedelta(minutes=buffer)

    # 매도가가 매수가보다 높은 경우 예측 결과를 전송합니다.
    if sell > buy:
        add_feed(mentor_page_id=mentor_page_id, username=username, password=password, coin=coin, buy=buy, sell=sell,
                 predict_minute_range=predict_minute_range, start_time=start_time)


if __name__ == '__main__':
    # 비트코인, 1분봉 데이터 사용
    # 15분 길이 예측, 예측 게시 버퍼 시간 3분
    mentor_page_id = "참가팀 별 멘토 ID"
    username = "참가팀 별 유저이름"
    password = "참가팀 별 비밀번호"
    coin = "KRW-BTC"
    minute = 1
    predict_minute_range = 15
    buffer = 3

    # 간단한 예시를 위해 최근 캔들 200개만 수집. 200개 이상의 캔들 수집 시 'to' 파라미터를 변경해가며 반복 수집하면 됩니다.
    # 초당 10회 이상 요청 시 수집에 실패할 수 있습니다. 자세한 내용은 업비트 OpenAPI Documentation 참조.
    utc_now = datetime.utcnow().replace(second=0, microsecond=0)
    candle = call_api(coin=coin, minute=minute, count=200, to=utc_now)
    candle = candle[::-1]

    # 종가 데이터 로그 차분
    close = candle["trade_price"].apply(np.log)

    # AR/MA 차수: 2, 차분 차수: 1인 ARIMA 모델 학습
    model = ARIMA(close.values, order=(2, 1, 2))
    model_fit = model.fit()
    print(model_fit.summary())

    # 스케줄러를 사용하여 1분마다 실시간 예측 함수 실행
    scheduler = BlockingScheduler()
    scheduler.add_job(send_prediction,
                      'cron',
                      args=[mentor_page_id, username, password, model_fit, coin, minute, predict_minute_range, buffer],
                      minute='*')
    scheduler.start()
