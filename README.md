# 가상화폐 분야 AI 모델링 공모전 세부 요강

## 개요
비트코인, 이더리움, 리플 등 업비트에 상장된 109개 가상화폐의 가격을 예측하는 AI 모델 개발 (109개 코인 중 자유 선택)

## 결과물 제출
제출 기간
1. 보고서/소스코드: 2022.02.21(월) ~ 2022.02.25(금)
2. 시뮬레이션: 2022.03.01(화) 00시 00분 ~ 2022.03.07(월) 23시 59분 기간 동안 실시간 가격 예측 결과를 API로 제출.(보고서 평가 통과 팀 대상)

제출 방법
1. 보고서/소스코드: 이메일 제출(joprr12@lab2ai.com)
2. 시뮬레이션: API 제출 가이드라인 참조.

제출 규격
1. 보고서: 한글(hwp), 워드(doc), 파워포인트(ppt) 중 택 1, 자유 양식.
2. 소스코드: 파이썬 스크립트 파일(py) 또는 소스코드가 게시된 Github Repository 주소.
3. 시뮬레이션: API 제출 가이드라인 참조.

## 제출 유의 사항
1. 보고서에는 전처리, 학습, 후처리, 추론 일련의 개발 과정을 상세히 기술하여야 함.
2. 소스코드의 실행 결과가 보고서 내용과 일치하여야 함.
3. 보고서/소스코드 제출 시 파이썬 및 라이브러리 버전 확인이 가능한 형태로 제출. (requirements.txt)
4. 실시간 예측 결과를 API 규격에 맞지 않게 전송할 경우 수익률 평가가 이루어지지 않을 수 있음. (감점)
5. API 예측 결과 전송이 원활히 이루어지고 있는지 확인이 필요한 경우 joprr12@lab2ai.com 으로 문의.

## 데이터
1. 본 공모전에서는 별도의 데이터가 제공되지 않으며, 참가자가 직접 필요한 데이터를 수집하여 사용.
2. 데이터 사용에 대한 제한은 없음.   ex) 바이낸스와 업비트 가격 차이 데이터를 모델 학습에 사용
3. 불안정한 데이터 소스 사용은 감점 요인이 될 수 있음.
4. 업비트 가격 데이터 수집: https://docs.upbit.com/reference/ 참조

## 모델링
1. 소스코드를 통해 AI 기술 활용 여부 및 재현 가능성 여부 검증 예정
2. 모델 학습/검증/최적화 단계에서는 수익률이 아닌 다른 평가 지표 사용 가능.   ex) 종가에 대한 RMSE
3. 모델의 예측 성능 평가 시 백테스팅을 통한 수익률 지표를 포함할 것을 권장.

## 예측
1. 가격 예측은 업비트에서 거래 가능한 109개 가상화폐의 원화 대비 가격 대상으로만 수행.

## 시뮬레이션 평가
1. 예측 당 평균 수익률로 평가. (수익률: 매도가÷매수가 / 예측 당 평균 수익률: 수익률 총합÷예측 횟수)
2. 예측의 길이(예측 종료 시간 - 예측 시작 시간)는 15분, 30분, 1시간, 3시간, 6시간, 12시간, 24시간 중에서 수행되어야 함. (그 외는 무효 처리)
3. 예측 길이의 50% 시점 이내 매수가 진입에 실패한 경우 해당 예측은 무효 처리. ex) 30분 길이의 예측은 15분 내 매수가 진입 성공하여야 함.
4. 매수가 진입에 성공하였으나 예측 종료 시간 전까지 매도가 도달에 실패한 경우 예측 종료 시점의 가격으로 매도 처리.
5. 예측 시작 시점이 예측 제출 시점 이전일 경우 무효 처리.
6. 동일 코인에 대해 예측 시간이 중복되는 경우 먼저 수행된 예측만 반영되며, 뒤에 이루어진 예측은 무효 처리.
7. 서로 다른 코인에 대한 예측은 예측 시간이 중복되어도 모두 수익률 계산에 반영됨.
8. 시뮬레이션 평가 기간 동안 최소 7회 이상의 예측을 수행하여야하며, 7회 미만으로 예측할 경우 0점 처리.

## 가이드라인

### 업비트 데이터 수집

dataset > upbit.py 참조

coin: 마켓 명 대문자 문자열 ex) 'KRW-BTC' (원화-비트코인)

minute: 수집하려는 캔들 분 단위

to: 마지막 캔들 시각. utc 시각 기준

count: 한 번에 가져올 캔들 개수. 최대 200개 캔들

```python
import pandas as pd
from datetime import datetime
from dataset.upbit import call_api

coin: str = 'KRW-BTC'
minute: int = 1
to: datetime = datetime.utcnow().replace(second=0, microsecond=0)
count: int = 10

candle_df: pd.DataFrame = call_api(coin=coin, minute=minute, to=to, count=count)
```

### 예측 결과 제출

api > submit.py 참조

mentor_page_id: 참가팀 별 id (안내 예정)

coin: 마켓 명 대문자 문자열. ex) 'KRW-BTC' (원화-비트코인)

buy: 매수가

sell: 매도가

predict_minute_range: 예측 길이(분)

start_time: 예측 시작 시점. kst 시각 기준

※예측 시작 시점은 반드시 제출 시점 이후여야 합니다.

※예측 소요시간 고려하여 예측 시작 시점은 1~5분 여유시간을 두는 것을 권장합니다. (main.py - buffer 변수 참조)

```python
from datetime import datetime
from api.submit import add_feed

mentor_page_id: str = '123456789'
coin: str = 'KRW-BTC'
buy: float = 51356000
sell: float = 51456000
predict_minute_range: int = 15
start_time: datetime = datetime(2022, 1, 9, 12, 5)

add_feed(mentor_page_id, coin, buy, sell, predict_minute_range, start_time)
```

### 데이터 수집, 모델 학습, 실시간 예측

main.py 참조

```python
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse
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
        add_feed(mentor_page_id=mentor_page_id, coin=coin, buy=buy, sell=sell,
                 predict_minute_range=predict_minute_range, start_time=start_time)


if __name__ == '__main__':
    # 비트코인, 1분봉 데이터 사용
    # 15분 길이 예측, 예측 게시 버퍼 시간 3분
    mentor_page_id = "123456789"
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
                      args=[mentor_page_id, model_fit, coin, minute, predict_minute_range, buffer],
                      minute='*')
    scheduler.start()
```

## 기타 문의사항
이메일 문의: joprr12@lab2ai.com
