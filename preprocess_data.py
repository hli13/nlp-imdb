import numpy as np
import os
import nltk
nltk.download('punkt')
import itertools
import io


# specify paths and directories
preproc_dir = './preprocessed_data'
imdb_dir = './aclImdb'

## create directory to store preprocessed data
if(not os.path.isdir(preproc_dir)):
    os.mkdir(preproc_dir)


def read_reviews(data):
    """
    Read train/test reviews and remove unwanted texts
    
    Parameters
    ----------
    data : str
        type of the data to be processed 
        option: train or test
        
    Returns
    -------
    x : list of list
        a list of tokenized reviews. 
        Each item contains one review that is tokenized and converted into a 
        list of tokens (i.e., words and symbols)
        
    """
    # get all of the training/testing reviews (including unlabeled reviews)
    data_dir = imdb_dir + '/' + data + '/'
    
    pos_filenames = []
    neg_filenames = []
    unsup_filenames = []
    
    # get a list of filenames for each category of reviews
    pos_filenames = os.listdir(data_dir + 'pos/')
    neg_filenames = os.listdir(data_dir + 'neg/')
    if (data == 'train'):
        unsup_filenames = os.listdir(data_dir + 'unsup/')
    
    # add the path to the directory in front of the filenames
    pos_filenames = [data_dir+'pos/'+filename for filename in pos_filenames]
    neg_filenames = [data_dir+'neg/'+filename for filename in neg_filenames]
    if (data == 'train'):
        unsup_filenames = [data_dir+'unsup/'+filename for filename in unsup_filenames]
    
    # concatenate all filenames together
    filenames = pos_filenames + neg_filenames + unsup_filenames
    
    # process each file:
    # - read the review
    # - remove unwanted texts, 
    # - tokenize each review
    # - save in the output
    count = 0
    x = []
    for filename in filenames:
        with io.open(filename,'r',encoding='utf-8') as f:
            line = f.readlines()[0]
        line = line.replace('<br />',' ')
        line = line.replace('\x96',' ')
        line = nltk.word_tokenize(line)
        line = [w.lower() for w in line]
        x.append(line)
        count += 1
    
    # print data info
    data = data + 'ing'
    print("\nNumber of %s reviews : %d" % (data, count))
    
    # number of tokens per review
    no_of_tokens = []
    for tokens in x:
        no_of_tokens.append(len(tokens))
    no_of_tokens = np.asarray(no_of_tokens)
    print("Total number of tokens : %d" % np.sum(no_of_tokens))
    print("Statistics of the number of tokens per review")
    print("Min  : %5d" % np.min(no_of_tokens))
    print("Max  : %5d" % np.max(no_of_tokens))
    print("Mean : %5d" % np.mean(no_of_tokens))
    print("Std  : %5d" % np.std(no_of_tokens))
    
    return x


def build_vocab(x_train, x_test):
    """
    Build dictionary/vacabulary based on the training dataset and save to files
    
    Parameters
    ----------
    x_train : list of list
        a list of tokenized training reviews. 
        Each item contains one review that is tokenized and converted into a 
        list of tokens (i.e., words and symbols)
    x_test : list of list
        a list of tokenized testing reviews. 
        Each item contains one review that is tokenized and converted into a 
        list of tokens (i.e., words and symbols)
        
    Returns
    -------
    None
    """
    # word_to_id and id_to_word
    # associate an id to every unique token in the training data
    all_tokens = itertools.chain.from_iterable(x_train)
    word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
    
    all_tokens = itertools.chain.from_iterable(x_train)
    id_to_word = [token for idx, token in enumerate(set(all_tokens))]
    id_to_word = np.asarray(id_to_word)
    
    ## let's sort the indices by word frequency instead of random
    x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
    count = np.zeros(id_to_word.shape)
    for x in x_train_token_ids:
        for token in x:
            count[token] += 1
    indices = np.argsort(-count)
    id_to_word = id_to_word[indices]
    count = count[indices]
    
    hist = np.histogram(count,bins=[1,10,100,1000,10000])
    print("\nHistogram of the overall token count :")
    print(hist)
    
    # the first 10 frequently used words/symbols
    print("\nThe first 10 frequently used words/symbols :")
    print("Word/Symbol \t Number of Occurances")
    for i in range(10):
        print("%7s \t %12d" % (id_to_word[i],count[i]))
    
    print("\nToken count for the top 100 frequently used words/symbols : %d [%.2f%%]" 
          % (np.sum(count[0:100]), 100.0*np.sum(count[0:100])/np.sum(count)))
    print("Token count for the top 8000 frequently used words/symbols : %d [%.2f%%]" 
          % (np.sum(count[0:8000]), 100.0*np.sum(count[0:8000])/np.sum(count)))
    
    ## recreate word_to_id based on sorted list
    word_to_id = {token: idx for idx, token in enumerate(id_to_word)}
    
    ## assign -1 if token doesn't appear in our dictionary
    ## add +1 to all token ids, we went to reserve id=0 for an unknown token
    x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
    x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]
    
    ## save dictionary
    dict_filename = preproc_dir + '/imdb_dictionary.npy'
    np.save(dict_filename,np.asarray(id_to_word))
    
    ## save training data to single text file
    train_filename = preproc_dir + '/imdb_train.txt'
    with io.open(train_filename,'w',encoding='utf-8') as f:
        for tokens in x_train_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")
    
    ## save test data to single text file
    test_filename = preproc_dir + '/imdb_test.txt'
    with io.open(test_filename,'w',encoding='utf-8') as f:
        for tokens in x_test_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")


def glove_embed(x_train, x_test):
    """
    Create word embedding with GloVe features and save to files
    
    Parameters
    ----------
    x_train : list of list
        a list of tokenized training reviews. 
        Each item contains one review that is tokenized and converted into a 
        list of tokens (i.e., words and symbols)
    x_test : list of list
        a list of tokenized testing reviews. 
        Each item contains one review that is tokenized and converted into a 
        list of tokens (i.e., words and symbols)
        
    Returns
    -------
    None
    """
    # GloVe feature        
    glove_filename = './glove.840B.300d.txt'
    with io.open(glove_filename,'r',encoding='utf-8') as f:
        lines = f.readlines()
    
    glove_dictionary = []
    glove_embeddings = []
    count = 0
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        glove_dictionary.append(line[0])
        embedding = np.asarray(line[1:],dtype=np.float)
        glove_embeddings.append(embedding)
        count+=1
        if(count>=100000):
            break
    
    glove_dictionary = np.asarray(glove_dictionary)
    glove_embeddings = np.asarray(glove_embeddings)
    # added a vector of zeros for the unknown tokens
    glove_embeddings = np.concatenate((np.zeros((1,300)),glove_embeddings))
    
    word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}
    
    x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
    x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]
    
    dict_filename_glove = preproc_dir + '/glove_dictionary.npy'
    embed_filename_glove = preproc_dir + '/glove_embeddings.npy'
    
    np.save(dict_filename_glove,glove_dictionary)
    np.save(embed_filename_glove,glove_embeddings)
    
    train_filename_glove = preproc_dir + '/imdb_train_glove.txt'
    with io.open(train_filename_glove,'w',encoding='utf-8') as f:
        for tokens in x_train_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")
    
    test_filename_glove = preproc_dir + '/imdb_test_glove.txt'
    with io.open(test_filename_glove,'w',encoding='utf-8') as f:
        for tokens in x_test_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")


if __name__ == "__main__":
    
    # read and tokenize training and testing datasets
    print("\nRead and Tokenize IMDb Reviews")
    print("-------------------------------")
    x_train = read_reviews('train')
    x_test = read_reviews('test')
    
    # build dictionary and save to files
    print("\nBuild Dictionary Based on the Training Reviews")
    print("-----------------------------------------------")
    build_vocab(x_train, x_test)
    
    # word embedding with GloVe features
    print("\nGenerate Word Embeddings with GloVe Features")
    print("--------------------------------------------")
    glove_embed(x_train, x_test)
    
    print("\nFinish Preprocessing Data")
    print("---------------------------")