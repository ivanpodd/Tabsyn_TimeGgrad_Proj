from numpy import nan, array, fromiter
from pandas import Series, DataFrame

employee_count_nm_categories = [
    'ДО 10', 
    'ОТ 11 ДО 50',
    'ОТ 11 ДО 30', 
    'ОТ 31 ДО 50',
    'ОТ 51 ДО 100', 
    'ОТ 101 ДО 500', 
    'БОЛЕЕ 500', 
    'ОТ 501 ДО 1000',
    'БОЛЕЕ 1001',  
    nan]

def encode_categorical(value, categories):
    item_converter = lambda item: categories.index(item)
    if isinstance(value, Series):
        return value.apply(item_converter)
    if isinstance(value, array):
        return fromiter((item_converter(item) for item in value))
    if isinstance(value, list):
        return [item_converter(item) for item in value]
    return categories.index(value)

def decode_categorical(index, categories):
    item_converter = lambda idx: categories[idx]
    if isinstance(index, Series):
        return index.apply(item_converter)
    if isinstance(index, array):
        return fromiter((item_converter(idx) for idx in index))
    if isinstance(index, list):
        return [item_converter(idx) for idx in index]
    return categories[index]

def format_TabSyn_embed(embedding):
    a, b, c = embedding.shape
    return embedding[:, 1:, :].reshape(a, (b-1)*c) #.flatten(dimensions=(1, 2))

def combine_transactions_users(transactions, users, on='user_id', how='left', drop=True):
    df = transactions[[on]].merge(users, how=how, on=on)
    if drop:
        df = df.drop(on, axis=1)
    return df

def convert_np_to_pd(emb, offset=0):
    a, b = emb.shape
    emb_df = DataFrame(data=emb,
                       columns=[f'dyn_real_{i}' for i in range(b)])
    # emb_df['timestamp'] = range(offset, offset+len(emb_df))
    emb_df.insert(0, 'timestamp', range(offset, offset+len(emb_df)))
    return emb_df

def combine_pd_np(df, emb):
    return df.join(convert_np_to_pd(emb)).drop(['user_id'], axis=1)
    
    
    