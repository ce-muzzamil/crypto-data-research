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

    def get_minimalistic_frame(
        self,
        at=80000,
        length=288,
        backwardlook=288,
        max_size=64,
        kinterval=5.0,
    ):
        aa = []
        size = 0
        look_past = 0
        mnis = []
        mxis = []
        end_at = at-look_past
        max_size += 1

        while size < max_size:
            start_at = at-length-look_past
            a = self.nf.iloc[start_at:end_at, :]
            mxis_, mnis_, ch = finfo(a, fee=0.005)
            mnis_ = np.array(mnis_)+start_at
            mxis_ = np.array(mxis_)+start_at
            mnis.extend(mnis_.tolist())
            mxis.extend(mxis_.tolist())
            a = a.assign(ch=0.0)
            a.loc[mxis_, 'ch'] = np.array(ch)*100.0

            aa.append(a)
            size = len(mxis)+len(mnis)
            look_past += backwardlook
            end_at = start_at

        a = pd.concat(aa).sort_index()
        indices = list(set(mnis+mxis))
        indices.sort()
        indices = indices[-max_size:]

        vectors = np.zeros((len(indices)-1, 7))

        an = a.loc[indices[0]:, :].to_numpy()
        nindices = np.array(indices)-indices[0]
        for i in range(len(indices)-1):
            vectors[i, [0, 2, 3]] = an[nindices[i]
                :nindices[i+1], [1, 4, 5]].mean(axis=0)
            stdp = an[nindices[i]:nindices[i+1], 1].std()
            if np.isnan(stdp):
                stdp = 0.0
            vectors[i, 1] = stdp
            vectors[i, [4, 5]] = an[nindices[i]                                    :nindices[i+1], [3, 6]].sum(axis=0)
            vectors[i, 6] = (nindices[i+1]-nindices[i])*kinterval

        a = a.loc[indices[1:], :]
        a.reset_index(drop=True, inplace=True)
        a = pd.concat([a.loc[:, ['stime', 'meanp', 'stdp', 'ch']], pd.DataFrame(
            vectors, columns=['avgp', 'stdavgp', 'taker', 'maker', 'vol', 'ntrds', 'gap'])], axis=1)

        index = a.pop('stime')
        a.set_index(pd.to_datetime(index, unit='ms'), drop=True, inplace=True)
        return a, np.array(mxis), np.array(mnis)

    def get_labled_frame_for_buy(
        self,
        at=80000,
        length=527,
        backwardlook=527,
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
                        x1, x2, y1, y2, y3 = self.get_labled_frame_for_buy(
                            at=ats[i])
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
