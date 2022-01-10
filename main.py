import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from apscheduler.schedulers.blocking import BlockingScheduler

from dataset.upbit import call_api
from api.submit import add_feed


def send_prediction(mentor_page_id, model_fit, coin, minute, predict_minute_range, buffer):
    """
    학습된 모델을 입력받아 실시간 예측을 수행하는 함수

    :param mentor_page_id: 참가팀 별 id
    :param model_fit: 학습된 모델
    :param coin: 코인 종류
    :param minute: 학습에 사용한 데이터의 분 봉 단위
    :param predict_minute_range: 예측 길이
    :param buffer: 예측 게시 버퍼 시간
    :return:
    """
    # 실시간 예측 결과 제출
    kst_now = datetime.now().replace(second=0, microsecond=0)
    utc_now = datetime.utcnow().replace(second=0, microsecond=0)

    recent_candle = call_api(coin, minute, count=1, to=utc_now)
    new_obs = recent_candle["trade_price"].apply(np.log)

    model_fit = model_fit.append(new_obs.values, refit=False)

    pred = model_fit.forecast(buffer + predict_minute_range)
    pred = np.exp(pred[buffer:])

    buy_index = np.argmin(pred)
    buy = min(pred)
    sell = max(pred[buy_index:])

    start_time = kst_now + timedelta(minutes=buffer)

    if sell > buy:
        add_feed(mentor_page_id=mentor_page_id, coin=coin, buy=buy, sell=sell,
                 predict_minute_range=predict_minute_range, start_time=start_time)


if __name__ == '__main__':
    # 비트코인, 1분봉 데이터 사용
    # 15분 길이 에측, 예측 게시 버퍼 시간 5분
    mentor_page_id = "123456789"
    coin = "KRW-BTC"
    minute = 1
    predict_minute_range = 15
    buffer = 5

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

    # 1분마다 예측 수행
    scheduler = BlockingScheduler()
    scheduler.add_job(send_prediction,
                      'cron',
                      args=[mentor_page_id, model_fit, coin, minute, predict_minute_range, buffer],
                      minute='*')
    scheduler.start()
