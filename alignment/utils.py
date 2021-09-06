from ignite.utils import apply_to_tensor
import torch
import numpy as np
import os

def get_grad_norm(parameters, norm_type =2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm +=param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm

def get_parameter_norm(parameters, norm_type = 2):
    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm +=param_norm ** norm_type
        total_norm = total_norm ** (1./ norm_type)
    except Exception as e:
        print(e)

    return total_norm

def apply_attention_make_batch(tensor,mask,index,tbtt_step,x_length,y_length):
    max_length = x_length // y_length
    #print(max_length)
    tensor_pad = torch.FloatTensor(tensor.size(0),tensor.size(1), tbtt_step * max(max_length))
    tensor_pad.zero_()

    mask_pad = torch.ones(tensor.size(0), tbtt_step * max(max_length))

    for i in range(tensor.size(0)):
        
        t = tensor[i,:,index[i]:index[i] + tbtt_step * max_length[i]]
        #print(t.shape)
        tensor_pad[i,:,:t.size(1)] = t

        m = mask[i,index[i]:index[i] + tbtt_step * max_length[i]]
        mask_pad[i,:m.size(0)] = m
        
    return tensor_pad,mask_pad.bool()

def y_make_batch(input_y_tensor,y_tensor,chunk_index,tbtt_step):
    y_input_pad = torch.LongTensor(input_y_tensor.size(0),tbtt_step)
    y_pad = torch.IntTensor(input_y_tensor.size(0),tbtt_step)

    for i in range(y_pad.size(0)):
        y_input = input_y_tensor[i,chunk_index[i]:chunk_index[i] + tbtt_step]
        y = y_tensor[i,chunk_index[i]:chunk_index[i] + tbtt_step]
        
        y_input_pad[i,:tbtt_step] = y_input
        y_pad[i,:tbtt_step] = y
        
    return y_input_pad,y_pad


def guided_attention( W,max_T,max_N): #w 54.68827160493827
    M = np.zeros((max_T, max_N), dtype=np.float32)
    for i in range(max_T):
        for j in range(max_N):
            if i * (max_N//max_T) - W <= j <= i * (max_N//max_T) + W:
                M[i,j] = 1
            else:
                M[i,j] = 0

    return M

def guided_attentions(size,W):
    bs,text_length,mel_length = size
    A = np.zeros((bs, text_length, mel_length), dtype=np.float32)
    for b in range(bs):
        A[b] = guided_attention(W,text_length,mel_length)

    #print('A',A.shape)
    
    return A
    
def detach_hidden(hidden):
    """Cut backpropagation graph.
    Auxillary function to cut the backpropagation graph by detaching the hidden
    vector.
    """
    return apply_to_tensor(hidden, torch.Tensor.detach)

'''def _guided_attention(N, max_N, T, max_T, g): #deepvoice
    W = np.zeros((max_N, max_T), dtype=np.float32)
    for n in range(N):
        for t in range(T):
            W[n, t] = 1 - np.exp(-(n / N - t / T)**2 / (2 * g * g))
    return W'''

def get_best_splits(a_prob):
    a_prob_ori = a_prob
    a_prob = a_prob.copy()
    a_prob[a_prob < 0.01] = 0
    f = np.zeros_like(a_prob)
    best_splits = np.zeros_like(a_prob, dtype=np.int)
    f[0] = np.cumsum(a_prob[0])
    for t in range(1, a_prob.shape[0]):
        prob_cumsum = np.cumsum(a_prob[t]) + 0.000001
        for s in range(t, a_prob.shape[1]):
            new_prob = f[t - 1, :s] + (prob_cumsum[s] - prob_cumsum[:s])
            new_prob[prob_cumsum[:s] / prob_cumsum[s] < 0.05] = 0
            best_f = new_prob.max()
            if best_f > 0:
                best_idx = np.where(new_prob == best_f)[0][-1]
                if new_prob[best_idx] >= f[t, s]:
                    f[t, s] = new_prob[best_idx]
                    best_splits[t, s] = best_idx

    route = [a_prob.shape[1] - 1]
    for i in range(a_prob.shape[0] - 1, 0, -1):
        route.append(best_splits[i, route[-1]])
    route.reverse()

    last_pos = 0
    total_scores = []
    for i in range(a_prob.shape[0]):
        total_scores.append(a_prob_ori[i, last_pos: route[i] + 1].sum())
        last_pos = route[i] + 1
    return np.array(route), np.array(total_scores)


def get_next_index(attn):
    ph_rel_end = attn.shape[0]
    try:
        splits_local, total_scores = get_best_splits(attn)
        if attn.argmax(1).max() > attn.shape[1] - 20:
            ph_rel_end = min(attn[:, -20:].argmax(0).min(), ph_rel_end)
        good_ali_ph = total_scores > 0.6
        cons_num = min(6, len(total_scores) - 2)
        if cons_num - 1 > 0:
            good_ali_ph_pad = np.pad(good_ali_ph, [cons_num - 1, 0], mode='constant', constant_values=False)
            good_ali_ph_cons = np.stack(
                [good_ali_ph_pad[x:-cons_num + x + 1] for x in range(cons_num - 1)] + [good_ali_ph], -1)
            good_ali_ph_cons = good_ali_ph_cons.all(1)
        else:
            good_ali_ph_cons = good_ali_ph
        good_ali_ph_cons_idxs = np.where(good_ali_ph_cons)[0]
        if len(good_ali_ph_cons_idxs) == 0:
            #retrain chunk
            return False
        else:
            ph_rel_end = min(good_ali_ph_cons_idxs[-1], ph_rel_end)
            return splits_local[ph_rel_end - 1]

    except Exception as e:
        print(e)


'''if __name__ == '__main__':
    print(guided_attention(6,20,20))
    #print(_guided_attention(5,5,5,5,0.2))'''