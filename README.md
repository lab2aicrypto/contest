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

## API 제출 가이드라인
mentor_page_id: 참가팀 별 id. (안내 예정)

coin: 마켓 명 대문자 문자열. ex) 'KRW-BTC' (원화-비트코인)

buy: 매수가

sell: 매도가

predict_minute_range: 예측 길이. (분)

start_time: 예측 시작 시점. (kst 시각)

※예측 시작 시점은 반드시 제출 시점 이후여야 합니다. 예측 소요시간 고려하여 예측 시작 시점은 1~5분 여유시간을 두는 것을 권장합니다. (main.py - buffer 변수 참조)

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

## 기타 문의사항
이메일 문의: joprr12@lab2ai.com
