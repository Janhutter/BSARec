import math

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set([actual[i]])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len([actual[user_id]]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set([actual[user_id]])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))

# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

## Fairness metrics

def unfairness(rec_df, or_df, group, weight_vector):
    """
    IAA (Inequity of Amortized Attention).

    Related Paper: "Amortizing Individual Fairness in Rankings"
    First Author:  Asia J. Biega

    Args:
        rec_df (pandas.DataFrame): complete set of rankings to measure
        or_df: relevance score 
        group (GroupInfo Object): contains group information from rec_df
        weight_vector (position based object): see position module

    Columns Used
        rating -- 0/1, whether a user clicked on an item (aka relevance)
    """
    
    nlists = rec_df['user'].nunique() #number of users
    
    # group to compute per-group exposure and relevance
    g_att = weight_vector.groupby(rec_df[group.category]).sum() / nlists
    g_rel = or_df['score'].groupby(or_df[group.category]).sum() / nlists
   
    #g_rel = or_df[group.category].sum() / nlists
    #print(g_rel)

    # the metric is the L1 norm
    return np.abs(g_att - g_rel).sum()