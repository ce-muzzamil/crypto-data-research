"""This is a database class"""

import json
from multiprocessing import Process
import tempfile
import glob

import numpy as np
import pandas as pd
from rise_and_fall import *


class Dataset:
    """ This a Dataset class"""

    def __init__(self):
        with open("datasplit.json", 'r') as file:
            dataset = json.load(file)

        self.validation_set = dataset['validation']
        self.test_set = dataset['testing']
        self.train_set = dataset['training']

        f = pd.read_csv("database/BTCUSDT.csv")
        nf = pd.DataFrame()
        nf["stime"] = f.loc[:, "0"]
        nf["meanp"] = f.loc[:, [str(i) for i in [1, 2, 3, 4]]].mean(axis=1)
        nf["stdp"] = f.loc[:, [str(i) for i in [1, 2, 3, 4]]].std(axis=1)
        nf["vol"] = f.loc[:, "7"]
        nf["taker"] = f.loc[:, "10"]/f.loc[:, "7"]
        nf["maker"] = (f.loc[:, "7"]-f.loc[:, "10"])/f.loc[:, "7"]
        nf.fillna(0.0, inplace=True)
        nf["ntrds"] = f.loc[:, "8"]
        self.mu = nf.mean()
        self.sigma = nf.std()
        self.mx = nf.max()
        self.mn = nf.min()
        self.nf = nf

    # def get_minimalistic_frame(
    #     self,
    #     at=1000,
    #     length=576,
    #     backwardlook=288,
    #     max_size=64,
    #     kinterval=5.0,
    # ):
    #     aa = []
    #     size = 0
    #     look_past = 0
    #     mn_index = np.zeros((0,))
    #     mx_index = np.zeros((0,))
    #     end_at = at-look_past
    #     max_size += 1
    #     last_vectors = []
        
    #     while size < max_size:
    #         start_at = at-length-look_past
    #         a = self.nf.iloc[start_at:end_at, :]
    #         a.reset_index(drop=True, inplace=True)

    #         mxis, mnis, ch = finfo(a, fee=0.005)
    #         mn_index = np.append(mn_index, np.array(mnis)+start_at)
    #         mx_index = np.append(mx_index, np.array(mxis)+start_at)

    #         a = a.assign(mnis=0.0)
    #         a = a.assign(mxis=0.0)
    #         a = a.assign(ch=0.0)
    #         a.loc[mnis, 'mnis'] = 1.0
    #         a.loc[mxis, 'mxis'] = 1.0
    #         a.loc[mxis, 'ch'] = np.array(ch)*100.0

    #         fltr = np.logical_or(a['mnis'], a['mxis'])
    #         indices = a[fltr].index
    #         vectors = np.zeros((indices.shape[0], 7))
    #         for i in range(indices.shape[0]-1):
    #             vectors[i+1, [0, 2, 3]] = a.loc[0 if i == 0 else indices[i]
    #                                                             :indices[i+1], ['meanp', 'taker', 'maker']].mean().to_numpy()
    #             stdp = a.loc[0 if i == 0 else indices[i]
    #                                          :indices[i+1], 'meanp'].std()
    #             if np.isnan(stdp):
    #                 stdp = 0.0
    #             vectors[i+1, 1] = stdp
    #             vectors[i+1, [4, 5]] = a.loc[0 if i == 0 else indices[i]
    #                 :indices[i+1], ['vol', 'ntrds']].sum()
    #             vectors[i+1, 6] = (indices[i+1]-indices[i])*kinterval

    #         last_vector = np.zeros((7,))
    #         last_vector[[0, 2, 3]] = a.loc[indices[-1]:,
    #                                        ['meanp', 'taker', 'maker']].mean().to_numpy()
    #         last_vector[1] = a.loc[indices[-1]:, 'meanp'].std()
    #         last_vector[[4, 5]] = a.loc[indices[-1]:, ['vol', 'ntrds']].sum()
    #         last_vector[6] = (a.index[-1]-indices[-1])*kinterval
    #         last_vectors.append(last_vector)

    #         a = a.loc[fltr, :]
    #         a.reset_index(drop=True, inplace=True)
    #         a = pd.concat([a.loc[:, ['stime', 'meanp', 'stdp', 'ch']], pd.DataFrame(
    #             vectors, columns=['avgp', 'stdavgp', 'taker', 'maker', 'vol', 'ntrds', 'gap'])], axis=1)
    #         aa.append(a)
    #         size += a.shape[0]
    #         look_past += backwardlook
    #         end_at = start_at

    #     for i in range(len(last_vectors)-1):
    #         aa[i].loc[0, ['avgp', 'stdavgp', 'taker', 'maker',
    #                       'vol', 'ntrds', 'gap']] = last_vectors[i+1]
    #     aa.reverse()
    #     a = pd.concat(aa)
    #     index = a.pop('stime')
    #     a.set_index(pd.to_datetime(index, unit='ms'), drop=True, inplace=True)

    #     return a.iloc[-(max_size-1):], mx_index, mn_index

    def get_minimalistic_frame(
        self,
        at=1000,
        length=144,
        backwardlook=144,
        max_size=64,
        kinterval=5.0,
    ):
        aa = []
        size = 0
        look_past = 0
        mn_index = np.zeros((0,))
        mx_index = np.zeros((0,))
        end_at = at-look_past
        max_size += 1
        last_vectors = []
        first_vectors = []

        while size < max_size:
            start_at = at-length-look_past
            a = self.nf.iloc[start_at:end_at, :]
            a.reset_index(drop=True, inplace=True)

            mxis, mnis, ch = finfo(a, fee=0.005)
            mn_index = np.append(mn_index, np.array(mnis)+start_at)
            mx_index = np.append(mx_index, np.array(mxis)+start_at)

            a = a.assign(mnis=0.0)
            a = a.assign(mxis=0.0)
            a = a.assign(ch=0.0)
            a.loc[mnis, 'mnis'] = 1.0
            a.loc[mxis, 'mxis'] = 1.0
            a.loc[mxis, 'ch'] = np.array(ch)*100.0

            fltr = np.logical_or(a['mnis'], a['mxis'])
            indices = a[fltr].index
            vectors = np.zeros((indices.shape[0], 7))

            first_vector = np.zeros((7,))
            first_vector[[0, 2, 3]] = a.loc[:indices[0],['meanp', 'taker', 'maker']].mean().to_numpy()
            first_vector[1] = a.loc[:indices[0], 'meanp'].std()
            first_vector[[4, 5]] = a.loc[:indices[0], ['vol', 'ntrds']].sum()
            first_vector[6] = (indices[0]-a.index[0])*kinterval
            first_vectors.append(first_vector)

            for i in range(indices.shape[0]-1):
                vectors[i+1, [0, 2, 3]] = a.loc[indices[i]:indices[i+1], ['meanp', 'taker', 'maker']].mean().to_numpy()
                stdp = a.loc[indices[i]:indices[i+1], 'meanp'].std()
                if np.isnan(stdp):
                    stdp = 0.0
                vectors[i+1, 1] = stdp
                vectors[i+1, [4, 5]] = a.loc[indices[i]:indices[i+1], ['vol', 'ntrds']].sum()
                vectors[i+1, 6] = (indices[i+1]-indices[i])*kinterval


            last_vector = np.zeros((7,))
            last_vector[[0, 2, 3]] = a.loc[indices[-1]:,['meanp', 'taker', 'maker']].mean().to_numpy()
            last_vector[1] = a.loc[indices[-1]:, 'meanp'].std()
            last_vector[[4, 5]] = a.loc[indices[-1]:, ['vol', 'ntrds']].sum()
            last_vector[6] = (a.index[-1]-indices[-1])*kinterval
            last_vectors.append(last_vector)

            a = a.loc[fltr, :]
            a.reset_index(drop=True, inplace=True)
            a = pd.concat([a.loc[:, ['stime', 'meanp', 'stdp', 'ch']], pd.DataFrame(
                vectors, columns=['avgp', 'stdavgp', 'taker', 'maker', 'vol', 'ntrds', 'gap'])], axis=1)
            aa.append(a)
            size += a.shape[0]
            look_past += backwardlook
            end_at = start_at

        for i in range(len(last_vectors)-1):
            aa[i].loc[0, ['avgp', 'stdavgp', 'taker', 'maker',
                        'vol', 'ntrds', 'gap']] = np.array([first_vectors[i],last_vectors[i+1]]).mean()
        aa.reverse()
        a = pd.concat(aa)
        index = a.pop('stime')
        a.set_index(pd.to_datetime(index, unit='ms'), drop=True, inplace=True)
        return a.iloc[-(max_size-1):], mx_index, mn_index

    def get_labled_frame_for_buy(
        self,
        at=80000,
        length=144,
        backwardlook=144,
        max_size=64,
        kinterval=5.0,
    ):
        a, mxis, _ = self.get_minimalistic_frame(
            at=at, length=length, backwardlook=backwardlook, max_size=max_size-1, kinterval=kinterval)
        mxi = int(mxis.max())
        mnis = []
        inc = 0
        while len(mnis) < 2:
            next_mxis, mnis, next_chs = finfo(
                self.nf.loc[mxi:mxi+48+inc], fee=0.002)
            inc += 12
        next_mni = mxi+mnis[0]
        next_mxi = mxi+next_mxis[0]
        next_ch = next_chs[0]
        if next_mni-mxi < 2:
            next_mni = mxi+mnis[1]
            next_mxi = mxi+next_mxis[0]
            next_ch = next_chs[1]
        random_mni = np.random.randint(mxi+1, next_mni)
        mni = (next_mni-random_mni)*kinterval
        next_mxi = (next_mxi-random_mni)*kinterval

        vector = np.zeros((a.shape[1],))
        ft = self.nf.iloc[mxi:random_mni, :]
        vector[[0, 1]] = ft.loc[ft.index[-1], ['meanp', 'stdp']]
        vector[[3, 5, 6]] = ft.loc[:, ['meanp', 'taker', 'maker']].mean()
        meanpstd = ft.loc[:, 'meanp'].std()
        if np.isnan(meanpstd):
            meanpstd = 0.0
        vector[4] = meanpstd
        vector[[7, 8]] = ft.loc[:, ['vol', 'ntrds']].sum()
        vector[9] = ft.shape[0]*kinterval
        ctime = pd.to_datetime(ft['stime'].iloc[-1], unit='ms')

        a = np.concatenate([a, vector.reshape(1, -1)], axis=0)
        ctime = ctime.weekday()*288 + (ctime.hour*60.0 + ctime.minute)/5.0
        return a, ctime, mni, next_mxi, next_ch*100.0

    def prepare_dataset(self, batch_size=1024, validation=False, test=False, buy=True):
        safety = 4096
        x1s = []
        x2s = []
        y1s = []
        y2s = []
        y3s = []
        if validation:
            ats = np.random.choice(self.validation_set,
                                   batch_size+safety).tolist()
        elif test:
            ats = np.random.choice(self.test_set, batch_size+safety).tolist()
        else:
            ats = np.random.choice(self.train_set, batch_size+safety).tolist()

        for i in range(len(ats)):
            if ats[i] not in ats[:i]:
                try:
                    if buy:
                        x1, x2, y1, y2, y3 = self.get_labled_frame_for_buy(at=ats[i])
                    else:
                        # x, y1, y2 = self.labeled_sell_frame(at=ats[i])
                        pass
                except:
                    continue
            else:
                continue

            x1s.append(x1)
            x2s.append(x2)
            y1s.append(y1)
            y2s.append(y2)
            if buy:
                y3s.append(y3)

            if len(x1s) >= batch_size:
                break

        x1s = np.array(x1s)
        x2s = np.array(x2s)
        y1s = np.array(y1s)
        y2s = np.array(y2s)
        if buy:
            y3s = np.array(y3s)
        if buy:
            return x1s, x2s, y1s, y2s, y3s
        else:
            # return xs, y1s, y2s
            pass

    def prepare_dataset_wrapper(self, folder='', index=0, batch_size=2**12, buy=True, validation=False, test=False):
        ds = self.prepare_dataset(
            batch_size=batch_size, buy=buy, validation=validation, test=test)
        if buy:
            x1, x2, y1, y2, y3 = ds
            np.save(folder+f'/x_{index}.npy', x1)
            np.save(folder+f'/y_{index}.npy', np.array([x2, y1, y2, y3]))
        else:
            # x, y1, y2 = ds
            # np.save(folder+f'/x_{index}.npy', x)
            # np.save(folder+f'/y_{index}.npy', np.array([y1, y2]))
            pass

    def get_data_set(self, batch_size, buy=True, validation=False, test=False, max_procs=10, single_process_size=2**8):
        x = []
        y = []
        with tempfile.TemporaryDirectory() as tmpdirname:

            required_procs = max(int(batch_size/single_process_size), 1)
            completed_procs = 0

            while completed_procs < required_procs:
                procs = []
                for i in range(min(max_procs, required_procs-completed_procs)):
                    index = completed_procs+i
                    proc = Process(target=self.prepare_dataset_wrapper, args=(
                        tmpdirname, index, single_process_size, buy, validation, test))
                    procs.append(proc)
                    proc.start()

                for proc in procs:
                    proc.join()

                completed_procs += len(procs)

            def retrive_index(r): return int(
                r.split("\\")[-1].split("_")[-1].split(".")[0])

            def retrive_name(r): return r.split("\\")[-1].split(".")[0]

            res = glob.glob(tmpdirname+'/*.*')

            x_files = [r for r in res if 'x' in retrive_name(r)]
            x_files_order = [retrive_index(r) for r in x_files]

            y_files = [r for r in res if 'y' in retrive_name(r)]
            ordered_y_files = []
            for order in x_files_order:
                ordered_y_files.extend(
                    [r for r in y_files if retrive_index(r) == order])

            x.extend([np.load(r) for r in x_files])
            y.extend([np.load(r) for r in ordered_y_files])

        x = np.concatenate(x)
        y = np.concatenate(y, axis=1)

        return x, [y[i] for i in range(y.shape[0])]
