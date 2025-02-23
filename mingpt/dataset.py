import itertools
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sys import getsizeof
class CharDataset(Dataset):
    def __init__(self, kings=False, test=False):
        extra_chars = []
        if kings:
            extra_chars += ['x', '-', 'K']
            # if test:
            #     file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_test.txt"
            # else:
            file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_version_2.txt"
            with open(file, "r") as f:
                if not test:
                    games = [s[:-2].replace('x', '#x#').replace('K', 'K#').replace('-', '#-#').replace(" ", "#").split("#") for s in tqdm(f.readlines()[:1000])]
                else:
                    games = [
                        s[:-2].replace('x', '#x#').replace('K', 'K#').replace('-', '#-#').replace(" ", "#").split("#")
                        for s in tqdm(f.readlines()[2_500_000:])]

        else:
            extra_chars += ['x', '-']
            if test:
                # file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_test.txt"
                pass
            else:
                file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_no_K.txt"
            with open(file, "r") as f:
                games = [s[:-2].replace('x', '#x#').replace('-', '#-#').replace(" ", "#").split("#") for s in tqdm(f.readlines())]
        chars = sorted([str(i) for i in range(1, 33)] + extra_chars) + [-100, ]
        data_size, vocab_size = len(games), len(chars)  # vocab size 61, with -100 sorted to the front
        max_len = max([len(games[_]) for _ in range(len(games))])  # should be 60 in Othello
        print('Dataset created has %d sequences, %d unique words.' % (data_size, vocab_size))
        
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = max_len
        self.block_size = 500  # for autoregressive training
        self.vocab_size = vocab_size
        # if hasattr(data, "ood_perc"):
        #     data.ood_perc = ood_perc  # turn on the randomness
        self.data = games
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]
        if len(chunk) <= self.block_size:
            chunk += [-100, ] * (self.block_size - len(chunk))  # -100 can be ignored in CE
        else:
            chunk = chunk[:self.block_size]
            # print(idx)
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        # print(x)
        # print(f'{getsizeof(x.data)/1024} KB')
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class CharDatasetNoMoveType(Dataset):
    def __init__(self, kings=False, test=False):
        extra_chars = []
        if kings:
            extra_chars += ['K']
            # if test:
            #     file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_test.txt"
            # else:
            file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_version_2.txt"
            with open(file, "r") as f:
                if not test:
                    games = [s[:-2].replace('x', '#').replace('K', 'K#').replace('-', '#').replace(" ", "#").split("#") for s in
                             tqdm(f.readlines()[:2_500_000])]
                    #TODO
                else:
                    games = [s[:-2].replace('x', '#').replace('K', 'K#').replace('-', '#').replace(" ", "#").split("#") for s in
                             tqdm(f.readlines()[2_500_000:])]

        else:
            extra_chars += []
            if test:
                # file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_test.txt"
                pass
            else:
                file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_no_K.txt"
            with open(file, "r") as f:
                games = [s[:-2].replace('x', '#').replace('-', '#').replace(" ", "#").split("#") for s in
                         tqdm(f.readlines())]
        chars = sorted([str(i) for i in range(1, 33)] + extra_chars) + [-100, ]
        data_size, vocab_size = len(games), len(chars)  # vocab size 61, with -100 sorted to the front
        max_len = max([len(games[_]) for _ in range(len(games))])  # should be 60 in Othello
        print('Dataset created has %d sequences, %d unique words.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = max_len
        self.block_size = 500  # for autoregressive training
        self.vocab_size = vocab_size
        # if hasattr(data, "ood_perc"):
        #     data.ood_perc = ood_perc  # turn on the randomness
        self.data = games

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]
        if len(chunk) <= self.block_size:
            chunk += [-100, ] * (self.block_size - len(chunk))  # -100 can be ignored in CE
        else:
            chunk = chunk[:self.block_size]
            # print(idx)
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next

        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.

        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        # print(x)
        # print(f'{getsizeof(x.data)/1024} KB')
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class CharDatasetWithSpaces(Dataset):
    def __init__(self, kings=False, test=False):
        extra_chars = []
        if kings:
            extra_chars += ['x', '-', 'K', ' ']
            # if test:
            #     file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_test.txt"
            # else:
            file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_version_2.txt"
            with open(file, "r") as f:
                if not test:
                    games = [
                        s[:-2].replace('x', '#x#').replace('K', 'K#').replace('-', '#-#').replace(" ", "#").split("#")
                        for s in tqdm(f.readlines()[:2_500_000])]
                else:
                    games = [
                        s[:-2].replace('x', '#x#').replace('K', 'K#').replace('-', '#-#').replace(" ", "#").split("#")
                        for s in tqdm(f.readlines()[2_500_000:])]

        else:
            extra_chars += ['x', '-', ' ']
            if test:
                # file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_test.txt"
                pass
            else:
                file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_no_K.txt"
            with open(file, "r") as f:
                games = [s[:-2].replace('x', '#x#').replace('-', '#-#').replace(" ", "# #").split("#") for s in
                         tqdm(f.readlines())]
        chars = sorted([str(i) for i in range(1, 33)] + extra_chars) + [-100, ]
        data_size, vocab_size = len(games), len(chars)  # vocab size 61, with -100 sorted to the front
        max_len = max([len(games[_]) for _ in range(len(games))])  # should be 60 in Othello
        print('Dataset created has %d sequences, %d unique words.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = max_len
        self.block_size = 500  # for autoregressive training
        self.vocab_size = vocab_size
        # if hasattr(data, "ood_perc"):
        #     data.ood_perc = ood_perc  # turn on the randomness
        self.data = games

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]
        if len(chunk) <= self.block_size:
            chunk += [-100, ] * (self.block_size - len(chunk))  # -100 can be ignored in CE
        else:
            chunk = chunk[:self.block_size]
            # print(idx)
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next

        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.

        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        # print(x)
        # print(f'{getsizeof(x.data)/1024} KB')
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class CharDatasetWithSpacesNoMoveType(Dataset):
    def __init__(self, kings=False, test=False):
        extra_chars = []
        if kings:
            extra_chars += ['K', ' ']
            # if test:
            #     file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_test.txt"
            # else:
            file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_version_2.txt"
            with open(file, "r") as f:
                if not test:
                    games = [
                        s[:-2].replace('x', '#x#').replace('K', 'K#').replace('-', '#-#').replace(" ", "#").split("#")
                        for s in tqdm(f.readlines()[:2_500_000])]
                else:
                    games = [
                        s[:-2].replace('x', '#x#').replace('K', 'K#').replace('-', '#-#').replace(" ", "#").split("#")
                        for s in tqdm(f.readlines()[2_500_000:])]

        else:
            extra_chars += [' ']
            if test:
                # file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_test.txt"
                pass
            else:
                file = "C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/datasets/generated_every_no_K.txt"
            with open(file, "r") as f:
                games = [s[:-2].replace('x', '#').replace('-', '#').replace(" ", "#").split("#") for s in
                         tqdm(f.readlines())]
        chars = sorted([str(i) for i in range(1, 33)] + extra_chars) + [-100, ]
        data_size, vocab_size = len(games), len(chars)  # vocab size 61, with -100 sorted to the front
        max_len = max([len(games[_]) for _ in range(len(games))])  # should be 60 in Othello
        print('Dataset created has %d sequences, %d unique words.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = max_len
        self.block_size = 500  # for autoregressive training
        self.vocab_size = vocab_size
        # if hasattr(data, "ood_perc"):
        #     data.ood_perc = ood_perc  # turn on the randomness
        self.data = games

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]
        if len(chunk) <= self.block_size:
            chunk += [-100, ] * (self.block_size - len(chunk))  # -100 can be ignored in CE
        else:
            chunk = chunk[:self.block_size]
            # print(idx)
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next

        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.

        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        # print(x)
        # print(f'{getsizeof(x.data)/1024} KB')
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

if __name__ == "__main__":
    with open("../datasets/generated_every_version_2.txt", "r") as f:
        a = f.readlines()
    # a = a[0][:-2].replace('x', '#x#').replace('K', '#K#').replace('-', '#-#').replace(" ", "#").split("#")
    # print(a)
    # a += [-100, ] * (1000 - len(a))
    # print(a)
    ds = CharDataset(True, False)
    print(ds[5])