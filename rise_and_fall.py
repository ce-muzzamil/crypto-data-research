import numpy as np
import pandas as pd

def ebi(s, i, ahead=12):  # earliest buying index
    s = np.copy(s)
    s = s[i:]

    mn = np.inf
    index = 0
    counter = 0
    is_started = False

    for j in range(s.shape[0]):
        if s[j] < mn:
            mn = s[j]
            index = j
            counter = 0
            is_started = True
        else:
            if is_started:
                counter += 1
        if counter > ahead:
            break
    return index+i


def lsi(s, i, ahead=12, fee=0.02):  # latest sell index
    s = np.copy(s)
    s = s[i:]

    mx = 0
    index = 0
    counter = 0
    is_started = False

    for j in range(s.shape[0]):
        if s[j] > fee and s[j] > mx:
            mx = s[j]
            index = j
            counter = 0
            is_started = True
        else:
            if is_started:
                counter += 1
        if counter > ahead:
            break
    return index+i, mx-fee


def quick_finfo(f: pd.DataFrame, fee=0.02, bahead=50, sahead=5):
    def change(f, i): return (f-i)/i
    ps = f.loc[:, "meanp"].to_numpy()
    n = ps.shape[0]
    ch_mat = np.zeros((n, n))
    for i in range(n):
        ch_mat[i, :] = change(ps, ps[i])
    ch_mat = np.triu(ch_mat)

    i = 0
    mxis = []
    mnis = []
    chs = []
    while i < n:
        bi = ebi(ch_mat[i, :], i, ahead=bahead)
        si, ch = lsi(ch_mat[bi, :], bi, ahead=sahead, fee=fee)
        if si > bi:
            mxis.append(si)
            mnis.append(bi)
            chs.append(ch)
        i = si+1
    return mxis, mnis, np.sum(chs)


def finfo(ff: pd.DataFrame, fee=0.02):
    def change(f, i): return (f-i)/i
    ps = ff.loc[:, "meanp"].to_numpy()
    n = ps.shape[0]
    ch_mat = np.zeros((n, n))
    for i in range(n):
        ch_mat[i, :] = change(ps, ps[i])
    ch_mat = np.triu(ch_mat)

    # creating filter with all buy sell pairs having hier than fee return
    filter = np.argwhere(ch_mat > fee)
    # checking whats the delay if first point is buy
    mxis = []
    mnis = []
    chs = []
    i = 0
    while i < ch_mat.shape[0]:
        x = filter[filter[:, 0] == i, 1].flatten()
        if x.shape[0] > 0:
            delay = x[0]  # delay = index of the potential sell
            # x_check = filter[filter[:,0]==i, 1]
            # cheking if a better point in the delay period
            x_ = filter[np.logical_and(np.logical_and(
                filter[:, 0] <= delay, filter[:, 1] <= delay), filter[:, 0] >= i), :]

            if x_.shape[0] > 0:
                i_ = x_[np.argmin(x_[:, 1])][0]
                if i != i_:
                    i = i_
                    continue

                x_ = x_[np.argmax(ch_mat[x_[:, 0], x_[:, 1]])]
                buyat = x_[0]
            else:
                # if a better point is not present in the delay period then buy at the initial check point
                buyat = i

            si, _ = lsi(ch_mat[buyat, :], buyat, ahead=3, fee=fee)
            # si = delay
            mnis.append(buyat)
            i = si+1
        else:
            i += 1

    # finding best sell as a max between two buys
    i = 1
    while i < len(mnis):
        mxi = np.argmax(ps[mnis[i-1]:mnis[i]])+mnis[i-1]
        chs.append(ch_mat[mnis[i-1], mxi])
        mxis.append(mxi)
        i += 1
    
    if len(mnis)>0 and len(mnis)>len(mxis):
        mnis.pop()
    return mxis, mnis, np.sum(chs)
