import pandas as pd

def prefilter_items(data):
    # Оставим только 5000 самых популярных товаров
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_5000 = popularity.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()
    #добавим, чтобы не потерять юзеров
    data.loc[~data['item_id'].isin(top_5000), 'item_id'] = 999999
    
    # Уберем 50 самых популярных
    data.loc[data['item_id'].isin(top_5000[:50]), 'item_id'] = 999999
    
    # Уберем самые непопулряные
    popularity_pu = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity_pu.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    popularity_pu['share_unique_users'] = popularity_pu['share_unique_users'] / data['user_id'].nunique()
    popularity_pu.sort_values('share_unique_users', ascending=False, inplace=True)
    unpop_list = popularity_pu[popularity_pu['share_unique_users'] < 0.01].item_id.tolist()
    data.loc[data['item_id'].isin(unpop_list), 'item_id'] = 999999
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    
    # Уберем не интересные для рекоммендаций категории (department)
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    
    # Уберем слишком дорогие товарыs
    
    # ...
    
    
    
    return data


def postfilter_items():
    pass
