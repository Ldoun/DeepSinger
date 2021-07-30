from ignite.utils import apply_to_tensor
import torch
import numpy as np

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
    
    last_index = []
    for i in range(tensor.size(0)):
        
        t = tensor[i,:,index[i]:index[i] + tbtt_step * max_length[i]]
        #print(t.shape)
        tensor_pad[i,:,:t.size(1)] = t

        m = mask[i,index[i]:index[i] + tbtt_step * max_length[i]]
        mask_pad[i,:m.size(0)] = m
        last_index.append(index[i] + tbtt_step * max_length[i])
        
    return tensor_pad,mask_pad.bool(),last_index



def guided_attention( W,max_N, max_T): #w 54.68827160493827
    M = np.zeros((max_N, max_T), dtype=np.float32)
    for i in range(max_N):
        for j in range(max_T):
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

if __name__ == '__main__':
    print(guided_attention(6,20,20))
    #print(_guided_attention(5,5,5,5,0.2))