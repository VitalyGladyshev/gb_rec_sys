import pandas as pd

def prefilter_items(data):
    # Оставим только 5000 самых популярных товаров
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_5000 = popularity.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()
    #добавим, чтобы не потерять юзеров
    data.loc[~data['item_id'].isin(top_5000), 'item_id'] = 999999
    
    # Уберем самые популярные 
    
    # Уберем самые непопулряные 
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    
    # Уберем не интересные для рекоммендаций категории (department)
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    
    # Уберем слишком дорогие товарыs
    
    # ...
    
    return data


def postfilter_items():
    pass
