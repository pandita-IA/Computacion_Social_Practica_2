import pandas as pd
import random
import numpy as np
import math
from sklearn.model_selection import train_test_split


def process_dataframe_according_to_item_and_user(df, n_items, n_users):

    """
        Filtramos por los items que hayan aparecido al menos n_items veces y
        los usuarios que hayan comprado al menos n_users de los items validos
    """
    # Convertir a tipos de datos adecuados
    df['rating'] = df['rating'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Filtrar por ítems que han sido valorados al menos n_items veces
    item_counts = df['item'].value_counts()
    valid_items = item_counts[item_counts > n_items].index
    df_filtered = df[df.item.isin(valid_items)]

    # Filtrar por usuarios que han comprado al menos n_users veces algún ítem válido
    user_counts = df_filtered['user'].value_counts()
    valid_users = user_counts[user_counts > n_users].index
    df_filtered = df_filtered[df_filtered.user.isin(valid_users)]

    # Ordenar por timestamp
    df_filtered.sort_values(by='timestamp', inplace=True)

    return df_filtered


def rating_average (ratings, NUM_ITEMS, u):
  acc = 0
  count = 0

  for i in range(NUM_ITEMS):
    if ratings[u][i] != None:
      acc += ratings[u][i]
      count += 1

  if count == 0:
    return None
  avg = acc / count

  return avg

def correlation_similarity(ratings, NUM_ITEMS, u, v):
  num = 0
  den_u = 0
  den_v = 0

  count = 0

  avg_u = rating_average(ratings, NUM_ITEMS, u)
  avg_v = rating_average(ratings, NUM_ITEMS, v)


  for i in range(NUM_ITEMS):
    if ratings[u][i] != None and ratings[v][i] != None:
      r_u = ratings[u][i]
      r_v = ratings[v][i]

      num += (r_u - avg_u) * (r_v - avg_v)
      den_u += (r_u - avg_u) * (r_u - avg_u)
      den_v += (r_v - avg_v) * (r_v - avg_v)

      count += 1

  if count > 0 and den_u != 0 and den_v != 0:
    cor = num / math.sqrt( den_u * den_v )
    return cor;
  else:
    return None
  

def jmsd_similarity (ratings, NUM_ITEMS, MAX_RATING, MIN_RATING, u, v):

  union = 0
  intersection = 0
  diff = 0

  for i in range(NUM_ITEMS):
    if ratings[u][i] != None and ratings[v][i] != None:
      r_u = (ratings[u][i] - MIN_RATING) / (MAX_RATING - MIN_RATING)
      r_v = (ratings[v][i] - MIN_RATING) / (MAX_RATING - MIN_RATING)

      diff = (r_u - r_v) * (r_u - r_v)

      intersection += 1
      union += 1

    elif ratings[u][i] != None or ratings[v][i] != None:
      union += 1


  if intersection > 0:
    jaccard = intersection / union
    msd = diff / intersection
    return jaccard * (1 - msd)
  else:
    return None
  

def get_neighbors (k, similarities):

  neighbors = [None for _ in range(k)]

  for n in range(k):

    max_similarity = 0
    neighbor = None

    for v, sim in enumerate(similarities):
      if v not in neighbors and sim != None and sim > max_similarity:
        max_similarity = sim
        neighbor = v

    neighbors[n] = neighbor

  return neighbors

def average_prediction (ratings, i, neighbors):
  acc = 0
  count = 0

  for n in neighbors:
    if n == None: break
    if ratings[n][i] != None:
      acc += ratings[n][i]
      count += 1

  if count > 0:
    prediction = acc / count
    return prediction
  else:
    return None

def weighted_average_prediction (ratings, i, neighbors, similarities):
  num = 0
  den = 0

  for n in neighbors:
    if n == None: break

    if ratings[n][i] != None:
      num += similarities[n] * ratings[n][i]
      den += similarities[n]

  if den > 0:
    prediction = num / den
    return prediction
  else:
    return None

def deviation_from_mean_prediction (ratings, NUM_ITEMS, u, i, neighbors):
  acc = 0
  count = 0

  for n in neighbors:
    if n == None: break

    if ratings[n][i] != None:
      avg_n = rating_average(ratings, NUM_ITEMS, n)
      acc += ratings[n][i] - avg_n
      count += 1

  if count > 0:
    avg_u = rating_average(ratings, NUM_ITEMS, u)
    prediction = avg_u + acc / count
    return prediction
  else:
    return None
  
def get_recommendations (N, predictions):
  recommendations = [None for _ in range(N)]

  for n in range(N):

    max_value = 0
    item = None

    for i, value in enumerate(predictions):
      if i not in recommendations and value != None and value > max_value:
        max_value = value
        item = i

    recommendations[n] = item

  return recommendations

def has_test_ratings (test_ratings, NUM_ITEMS, u):
  for i in range(NUM_ITEMS):
    if test_ratings[u][i] != np.nan:
      return True
  return False

def get_user_mae (test_ratings, NUM_ITEMS, u, predictions):
  mae = 0
  count = 0

  for i in range(NUM_ITEMS):
    if test_ratings[u][i] != None and predictions[u][i] != None:
      mae += abs(test_ratings[u][i] - predictions[u][i])
      count += 1

  if count > 0:
    return mae / count
  else:
    return None
  
def get_mae (test_ratings, NUM_ITEMS, NUM_USERS, predictions):
  mae = 0
  count = 0

  for u in range(NUM_USERS):
    if has_test_ratings (test_ratings, NUM_ITEMS, u):
      user_mae = get_user_mae(test_ratings, NUM_ITEMS, u, predictions)

      if user_mae != None:
        mae += user_mae
        count += 1


  if count > 0:
    return mae / count
  else:
    return None
  

def get_user_rmse (test_ratings, NUM_ITEMS, u, predictions):
  mse = 0
  count = 0

  for i in range(NUM_ITEMS):
    if test_ratings[u][i] != None and predictions[u][i] != None:
      mse += (test_ratings[u][i] - predictions[u][i]) * (test_ratings[u][i] - predictions[u][i])
      count += 1

  if count > 0:
    return math.sqrt(mse / count)
  else:
    return None
  
def get_rmse (test_ratings, NUM_ITEMS, NUM_USERS, predictions):
  rmse = 0
  count = 0

  for u in range(NUM_USERS):
    if has_test_ratings (test_ratings, NUM_ITEMS, u):
      user_rmse = get_user_rmse(test_ratings, NUM_ITEMS, u, predictions)

      if user_rmse != None:
        rmse += user_rmse
        count += 1


  if count > 0:
    return rmse / count
  else:
    return None
  

def get_user_precision (test_ratings, theta, u, predictions, N):
  precision = 0
  count = 0
  recommendations = get_recommendations(N, predictions[u])

  for i in recommendations:
    if i != None and test_ratings[u][i] != None:
      precision += 1 if test_ratings[u][i] >= theta else 0
      count += 1

  if count > 0:
    return precision / count
  else:
    return None
  
def get_precision(test_ratings, NUM_ITEMS, NUM_USERS, theta, predictions, N):
  precision = 0
  count = 0

  for u in range(NUM_USERS):
    if has_test_ratings (test_ratings, NUM_ITEMS, u):
      user_precision = get_user_precision(test_ratings, theta, u, predictions, N)

      if user_precision != None:
        precision += user_precision
        count += 1


  if count > 0:
    return precision / count
  else:
    return None
  

def get_user_recall (test_ratings, NUM_ITEMS, theta, u, predictions, N):
  recall = 0
  count = 0
  recommendations = get_recommendations(N, predictions[u])

  for i in range(NUM_ITEMS):
    if test_ratings[u][i] != None and predictions[u][i] != None:
      if test_ratings[u][i] >= theta:
        recall += 1 if i in recommendations else 0
        count += 1

  if count > 0:
    return recall / count
  else:
    return None
  
def get_recall (test_ratings, NUM_ITEMS, NUM_USERS,theta, predictions, N):
  recall = 0
  count = 0

  for u in range(NUM_USERS):
    if has_test_ratings(test_ratings, NUM_ITEMS, u):
      user_recall = get_user_recall(test_ratings, NUM_ITEMS, theta, u, predictions, N)

      if user_recall != None:
        recall += user_recall
        count += 1


  if count > 0:
    return recall / count
  else:
    return None
  

def get_user_f1 (test_ratings, NUM_ITEMS, theta, u, predictions, N):
  precision = get_user_precision(test_ratings, theta, u, predictions, N)
  recall = get_user_recall(test_ratings, NUM_ITEMS, theta, u, predictions, N)

  if precision == None or recall == None:
    return None
  elif precision == 0 and recall == 0:
    return 0
  else:
    return 2 * precision * recall / (precision + recall)
  
  
def get_f1 (test_ratings, NUM_ITEMS, NUM_USERS, predictions, theta, N):
  f1 = 0
  count = 0

  for u in range(NUM_USERS):
    if has_test_ratings (test_ratings, NUM_ITEMS, u):
      user_f1 = get_user_f1(test_ratings, NUM_ITEMS, theta, u, predictions, N)

      if user_f1 != None:
        f1 += user_f1
        count += 1


  if count > 0:
    return f1 / count
  else:
    return None
  

def get_ordered_test_items(test_ratings, u):
  num_items = sum(x is not None for x in test_ratings[u])
  items = [None for _ in range(num_items)]

  for n in range(num_items):

    max_value = 0
    item = None

    for i,value in enumerate(test_ratings[u]):
      if i not in items and value != None and value > max_value:
        max_value = value
        item = i

    items[n] = item

  return items

def get_user_idcg (test_ratings, u):
  items = get_ordered_test_items(test_ratings, u)
  idcg = 0

  for pos, i in enumerate(items):
    idcg += (2 ** test_ratings[u][i] - 1) / math.log(pos+2, 2)

  return idcg

def get_user_dcg (test_ratings, u, recommendations):
  dcg = 0

  for pos, i in enumerate(recommendations):
    if i != None and test_ratings[u][i] != None:
      dcg += (2 ** test_ratings[u][i] - 1) / math.log(pos+2, 2)

  return dcg

def get_user_ndcg (test_ratings, u, predictions, N):
  recommendations = get_recommendations(N, predictions[u])
  dcg = get_user_dcg(test_ratings, u, recommendations)
  idcg = get_user_idcg(u)
  if idcg == 0:
    return 0
  else:
    return dcg / idcg
  
def get_ndcg (test_ratings, NUM_ITEMS, NUM_USERS, predictions):
  ndcg = 0
  count = 0

  for u in range(NUM_USERS):
    if has_test_ratings (test_ratings, NUM_ITEMS, u):
      user_ndcg = get_user_ndcg(test_ratings, u, predictions)

      if user_ndcg != None:
        ndcg += user_ndcg
        count += 1


  if count > 0:
    return ndcg / count
  else:
    return None
  

def procesar_dataframe(ruta_archivo, n_items, n_users, test_size=0.5):
  df_sports = pd.read_csv(ruta_archivo, names = ['item','user','rating','timestamp'])
  df_sports = process_dataframe_according_to_item_and_user(df_sports, n_items, n_users)

  df_sports['user'], user_codes = pd.factorize(df_sports['user'])
  df_sports['item'], item_codes = pd.factorize(df_sports['item'])

  NUM_ITEMS = len(item_codes)
  NUM_USERS = len(user_codes)

  ratings_todos = [[None for _ in range(NUM_ITEMS)] for _ in range(NUM_USERS)]


  for i, u, rating in df_sports[['item', 'user', 'rating']].itertuples(index=False):
      ratings_todos[u][i] = rating


  df_todos = pd.DataFrame(ratings_todos)

  # Reordenamos las columnas de forma aleatoria
  df_todos = df_todos.sample(frac=1, axis=1, random_state=42)
  

  # Partimos el dataset para lograr que ratings y test_ratings tengan el mismo tamaño
  # En lo unico en que difieren es en la asignación de None, en entrenameinto la parte de test está asignado con None y en test justamente lo contrario
  train_data, test_data = train_test_split(df_todos, test_size=test_size)

  test_data_copy = pd.concat([train_data.copy(), test_data.copy()])
  test_data_copy[:] = None

  num_users_for_test = len(test_data)
  train_data = pd.concat([train_data, test_data.copy()])


  train_data.iloc[num_users_for_test, NUM_ITEMS//2:] = None
  try:
      train_data.iloc[num_users_for_test:, :NUM_ITEMS//2] = test_data.iloc[:, :NUM_ITEMS//2]
  except:
      train_data.iloc[num_users_for_test-1:, :NUM_ITEMS//2] = test_data.iloc[:, :NUM_ITEMS//2]

  test_data_copy.iloc[:, NUM_ITEMS//2:] = test_data.iloc[:, NUM_ITEMS//2:]


  # Convertir los DataFrames en listas de listas con valores NaN reemplazados por None
  ratings = train_data.where(pd.notna(train_data), None).values.tolist()
  test_ratings = test_data_copy.where(pd.notna(test_data_copy), None).values.tolist()

  # Teniamos valores nan y none mezclados y para mantenerlo en el mismo formato nos decantamos con nones
  ratings = [[valor if valor in (1, 2, 3, 4, 5) else None for valor in lista] for lista in ratings]
  test_ratings = [[valor if valor in (1, 2, 3, 4, 5) else None for valor in lista] for lista in test_ratings]


  return ratings, test_ratings, NUM_ITEMS, NUM_USERS


def get_metricas(test_ratings, NUM_ITEMS, NUM_USERS, predictions, theta, N):
  print("MAE = ", get_mae(test_ratings, NUM_ITEMS, NUM_USERS, predictions))
  print("RMSE = ", get_rmse(test_ratings, NUM_ITEMS, NUM_USERS, predictions))
  print("Precision = ", get_precision(test_ratings, NUM_ITEMS, NUM_USERS, theta, predictions, N))
  print("Recall = ", get_recall(test_ratings, NUM_ITEMS, NUM_USERS,theta, predictions, N))
  print("F1 = ", get_f1(test_ratings, NUM_ITEMS, NUM_USERS, predictions, theta, N))


def compute_prediction (p_u, q_i, NUM_FACTORS):
  prediction = 0
  for k in range(NUM_FACTORS):
    prediction += p_u[k] * q_i[k]
  return prediction


def compute_biased_prediction (avg, b_u, b_i, p_u, q_i, NUM_FACTORS):
  deviation = 0
  for k in range(NUM_FACTORS):
    deviation += p_u[k] * q_i[k]

  prediction = avg + b_u + b_i + deviation
  return prediction