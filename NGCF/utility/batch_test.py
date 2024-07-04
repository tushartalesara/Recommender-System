import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import heapq

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = args.Ks

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = Ks
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r

def get_performance(user_pos_test, r, Ks):
    recall, ndcg = [], []

    recall.append(metrics.recall_at_k(r, Ks, len(user_pos_test)))
    ndcg.append(metrics.ndcg_at_k(r, Ks))

    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))
    r = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, Ks)


def test(model, users_to_test, drop_flag=False):
    result = {'recall': np.zeros(1), 'ndcg': np.zeros(1)}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        item_batch = range(ITEM_NUM)

        if drop_flag == False:
            u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                            item_batch,
                                                            [],
                                                            drop_flag=False)
            rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
        else:
            u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                            item_batch,
                                                            [],
                                                            drop_flag=True)
            rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users

    assert count == n_test_users
    pool.close()
    return result
