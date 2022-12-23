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
        self.nf = nf
        

    def get_processed_frame(
        self,
        f: pd.DataFrame,
        absolute_last_mni=None,
        length=288,
        kinterval=5.0,
        maxtime=18.0, #maximum time in a day e.g 288*5 mins = 18.0 where 18.0 is a representation
        maxtrds=5000.0,
        maxstdp=50.0,
        maxprice=5e4,
        maxvol=5e6,
        fee=0.005,
    ):
        def change(f, i): return (f-i)/i
        def normalize(x, xmx, xmn): return (x-xmn)/(xmx-xmn)

        frame = f.copy(deep=True)
        at = frame.index[-1]+1

        frame.reset_index(drop=True, inplace=True)
        mxis, mnis, chs = finfo(frame, fee=fee)

        if absolute_last_mni is not None:
            mni = absolute_last_mni - (at-length)
            mnis.append(mni)

        frame['stime'] = pd.to_datetime(frame['stime'], unit='ms')
        
        day = frame["stime"].apply(
            lambda x: x.weekday())  # mon=0, sun=6

        frame['stime'] = frame["stime"].apply(lambda x: (
            maxtime/length)*(x.hour*60.0+x.minute)/kinterval)

        frame['stime'] = frame['stime'] + day*18.0

        frame['ntrds'] = frame["ntrds"].apply(lambda x: x/maxtrds)
        frame['mnis'] = 0
        frame.loc[mnis, 'mnis'] = 1
        frame['mxis'] = 0
        frame.loc[mxis, 'mxis'] = 1
        frame['aqch'] = 0.0
        frame.loc[mxis, 'aqch'] = np.array(chs)*100.
        ps = frame.loc[:, 'meanp'].to_numpy()
        frame['ch'] = 0.0
        frame.loc[1:, 'ch'] = change(ps[1:], ps[:-1])*100.
        frame['stdp'] = frame["stdp"].apply(lambda x: x/maxstdp)
        frame['meanp'] = normalize(frame['meanp'], maxprice, 1e-9)*9 + 1
        frame['vol'] = normalize(frame['vol'], maxvol, 0.0)
        frame = frame.loc[:, ['stime', 'meanp', 'stdp', 'ch', 'mnis',
                              'mxis', 'aqch', 'taker', 'maker', 'vol', 'ntrds']]
        return frame

    def labeled_buy_frame(
        self,
        at=80000,
        length=288,
        kinterval=5.0,
        maxtime=18.0,
        fee=0.005,
        to_numpy=True
    ):
        frame = self.nf.iloc[at-length:at, :]  # nf is the database
        mxis1, mnis1, ch1 = finfo(frame, fee=fee)
        def change(f, i): return (f-i)/i
        buytime = pd.to_datetime(
            frame.loc[at-length+mnis1[-1], 'stime'], unit='ms')
        buyat = (maxtime/length)*(buytime.hour*60.0+buytime.minute)/kinterval
        selltime = pd.to_datetime(
            frame.loc[at-length+mxis1[-1], 'stime'], unit='ms')
        sellat = (maxtime/length)*(selltime.hour *
                                   60.0+selltime.minute)/kinterval
        potential_change = change(
            frame.loc[at-length+mxis1[-1], 'meanp'], frame.loc[at-length+mnis1[-1], 'meanp'])

        nat = np.random.randint(at-length+mxis1[-2], at-length+mnis1[-1])
        frame = self.nf.iloc[nat-length:nat, :]
        frame = self.get_processed_frame(
            frame, length=length, kinterval=kinterval, maxtime=maxtime, fee=fee)
        if to_numpy:
            return frame.to_numpy(), buyat, sellat, potential_change
        else:
            return frame, buyat, sellat, potential_change

    def labeled_sell_frame(
        self,
        at=80000,
        length=288,
        kinterval=5.0,
        maxtime=18.0,
        fee=0.005,
        to_numpy=True
    ):
        frame = self.nf.iloc[at-length:at, :]  # nf is the database
        mxis1, mnis1, ch1 = finfo(frame, fee=fee)
        def change(f, i): return (f-i)/i

        selltime = pd.to_datetime(
            frame.loc[at-length+mxis1[-1], 'stime'], unit='ms')
        sellat = (maxtime/length)*(selltime.hour *
                                   60.0+selltime.minute)/kinterval
        potential_change = change(
            frame.loc[at-length+mxis1[-1], 'meanp'], frame.loc[at-length+mnis1[-1], 'meanp'])

        nat = np.random.randint(at-length+mnis1[-1], at-length+mxis1[-1])

        frame = self.nf.iloc[nat-length:nat, :]
        frame = self.get_processed_frame(
            frame, length=length, kinterval=kinterval, maxtime=maxtime, fee=fee, absolute_last_mni=at-length+mnis1[-1])
        if to_numpy:
            return frame.to_numpy(), sellat, potential_change
        else:
            return frame, sellat, potential_change

    def prepare_dataset(self, batch_size=1024, validation=False, test=False, buy=True):
        safety = 4096
        xs = []
        y1s = []
        y2s = []
        y3s = []
        if validation:
            ats = np.random.choice(self.validation_set,
                                   batch_size+safety).tolist()
        elif test:
            ats = np.random.choice(self.test_set, batch_size+safety).tolist()
        else:
            ats = np.random.choice(self.test_set, batch_size+safety).tolist()

        for i in range(len(ats)):
            if ats[i] not in ats[:i]:
                try:
                    if buy:
                        x, y1, y2, y3 = self.labeled_buy_frame(at=ats[i])
                    else:
                        x, y1, y2 = self.labeled_sell_frame(at=ats[i])
                except:
                    continue
            else:
                continue

            xs.append(x)
            y1s.append(y1)
            y2s.append(y2)
            if buy:
                y3s.append(y3)

            if len(xs) >= batch_size:
                break

        xs = np.array(xs)
        y1s = np.array(y1s)
        y2s = np.array(y2s)
        if buy:
            y3s = np.array(y3s)
        if buy:
            return xs, y1s, y2s, y3s
        else:
            return xs, y1s, y2s

    def prepare_dataset_wrapper(self, folder='', index=0, batch_size=2**12, buy=True, validation=False, test=False):
        ds = self.prepare_dataset(
            batch_size=batch_size, buy=buy, validation=validation, test=test)
        if buy:
            x, y1, y2, y3 = ds
            np.save(folder+f'/x_{index}.npy', x)
            np.save(folder+f'/y_{index}.npy', np.array([y1, y2, y3]))
        else:
            x, y1, y2 = ds
            np.save(folder+f'/x_{index}.npy', x)
            np.save(folder+f'/y_{index}.npy', np.array([y1, y2]))

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
