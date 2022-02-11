def trim(price):
    if price < 0.1:
        price = round(price, 4)
    elif price < 1:
        price = round(price, 3)
    elif price < 10:
        price = round(price, 2)
    elif price < 100:
        price = round(price, 1)
    elif price < 1000:
        price = round(price, 0)
    elif price < 10000:
        price = (price // 5 * 5) + ((price % 5) // 2.5) * 5
    elif price < 100000:
        price = round(price, -1)
    elif price < 500000:
        price = (price // 50 * 50) + ((price % 50) // 25) * 50
    elif price < 1000000:
        price = round(price, -2)
    elif price < 2000000:
        price = (price // 500 * 500) + ((price % 500) // 250) * 500
    elif price >= 2000000:
        price = round(price, -3)

    return price
