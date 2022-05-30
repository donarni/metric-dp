import numpy as np

from annoy import AnnoyIndex

class metricDP():
   
    def __init__(self, vocabular, embedding, start_from=999):

        '''
        Code in part from Amazon SageMaker, Vocabular [Dictionary] is a token to
        index mapping, Embedding [Array] including special tokens such as [UNK],
        [PAD], [CLS], [SEP], [MASK], or [unused...]. Code expects special tokens
        at the front and regular tokens continuing from 'start_from'. Parameters
        defaulted to BERT (base, uncased).
        '''
				
        self.vocabular = vocabular
        self.embedding = embedding

        self.vocab_size = embedding.shape[0]
        self.embed_dim = embedding.shape[1]

        self.start_from = start_from

    def build_ann(self, metric='euclidean', n_trees=50):

        ''' Build Approximate Nearest Neighbors, excluding special tokens '''
        
        self.ann = AnnoyIndex(self.embed_dim, metric)

        for index, vector in enumerate(self.embedding[self.start_from:,:]):
            self.ann.add_item(index, vector)
            
        self.ann.build(n_trees)
        
    def privatize(self, tokens, epsilon=10, special_tokens=[0,100,101,102,103]):

        ''' Privatize a numeralized text, preserving special tokens. '''

        def replace(token, epsilon):
        
              random_vec = np.random.normal(size=self.embed_dim)
              normalized_vec = random_vec / np.linalg.norm(random_vec)
              magnitude = np.random.gamma(shape=self.embed_dim, scale=1/epsilon)
              noise = normalized_vec * magnitude
              original_vec = self.embedding[token]
              noisy_vector = original_vec + noise
              return self.ann.get_nns_by_vector(noisy_vector, 1)[0]

        assert self.ann != None, 'Build or Init ANNs.'

        for index, token in enumerate(tokens):
            assert token <= self.vocab_size, 'OOV'
            if token not in special_tokens:
                #token -= self.start_from
                token = replace(token, epsilon)
                tokens[index] = token + self.start_from
        return tokens
