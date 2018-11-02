

def init_embedding(emb):
    """Weight initialization of embeddings (in place).
    Best practise from fastai
    
    Arguments:
        emb {torch.nn.Embedding} -- Embedding
    """
    w = emb.weight.data
    sc = 2 / (w.shape[1]+1)
    w.uniform_(-sc, sc)